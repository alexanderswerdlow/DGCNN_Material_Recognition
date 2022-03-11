import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import sklearn.metrics as metrics
import os.path
from util import criterion, run_training, get_dataset_dir, get_data_dir
from fusion.networks import GraphNetwork
import torchvision.transforms as transforms
import h5py
from geomat import GeoMat
from tqdm import tqdm
import timm

pre_transform = T.NormalizeScale()

transforms3d = {"crop": 0,
                "dropout": 0.2,
                "rotate": 0,
                "mirror": 0}

train_dataset = GeoMat(get_dataset_dir(), True, None, pre_transform, transforms3d=transforms3d)
test_dataset = GeoMat(get_dataset_dir(), False, None, pre_transform)

# add transforms (flip, crop, dropout)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=6)


def train():
    model.train()

    train_loss, train_pred, train_true = 0, [], []
    for data in tqdm(train_loader):
        optimizer.zero_grad()

        feat = torch.stack([x_ for x_ in data.hha])
        points = torch.stack([x_ for x_ in data.point_cloud_3d])
        input = Data(pos=points, x=feat)

        out = model(input)
        loss = criterion(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        preds = out.max(dim=1)[1]
        train_loss += loss.item() * data.num_graphs
        train_true.append(data.y.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return train_loss / len(train_dataset), metrics.accuracy_score(train_true, train_pred), metrics.balanced_accuracy_score(train_true, train_pred)


def test():
    model.eval()
    correct = 0
    for data in test_loader:
        with torch.no_grad():
            feat = torch.stack([x_ for x_ in data.hha])
            points = torch.stack([x_ for x_ in data.point_cloud_3d])
            input = Data(pos=points, x=feat)

            pred = model(input).max(dim=1)[1]
        correct += pred.eq(data.y.to(device)).sum().item()
    return correct / len(test_loader.dataset)


class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def save_h5_features(fname_h5_feat3d, data_key, data_value):
    h5_feat = h5py.File(fname_h5_feat3d.strip(), "a")
    h5_feat.create_dataset(data_key, data=data_value, compression="gzip", compression_opts=4, dtype="float")
    h5_feat.close()


@torch.no_grad()
def extract_features(loader, loader_name, nlayer=18):
    model.eval()
    model.obtain_intermediate(nlayer)

    features = SaveFeatures(list(model.children())[nlayer])

    for _, data in enumerate(loader):

        feat = torch.stack([x_ for x_ in data.hha])
        points = torch.stack([x_ for x_ in data.point_cloud_3d])
        input = Data(pos=points, x=feat)

        graph = model(input)
        feat = features.features

        print(feat.shape)
        x1,x2 = None

        for i in range(0, x1.size(0)):

            # TODO
            feature = x2[i].view(x2.size(1), -1).permute(1, 0)

            points_average = torch.nn.AvgPool2d(8, count_include_pad=False, ceil_mode=False)
            points_torch_32 = points_average(torch.tensor(data.img_point_cloud[i]).permute(2, 0, 1))
            points_d32 = np.reshape(points_torch_32.data.numpy(), (-1, 3))

            save_name = f"{get_data_dir()}/fusion/3d/{data.dataset_idx[i]}.h5"
            save_h5_features(save_name, "features", feature.detach().cpu().numpy())
            save_h5_features(save_name, "points", points_d32)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf_3d = 'multigraphconv_9_16_0,b_0,r_0,pnv_max_0.05_0,multigraphconv_9_16_0,b_0,r_0,pnv_max_0.08_0,multigraphconv_9_32_0,b_0,r_0,pnv_max_0.12_0,multigraphconv_9_64_0,b_0,r_0,pnv_max_0.24_0,multigraphconv_9_128_1,b_1,r_1,gp_avg_1,d_0.2_1,f_19_cp_1'

    model = GraphNetwork(config=conf_3d, nfeat=3)

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model_name = os.path.basename(__file__).rstrip(".py")
    run_training(model_name, train, test, model, optimizer, scheduler, total_epochs=3)
    print("done")
    extract_features(train_loader, 'train')
    extract_features(test_loader, 'test')
