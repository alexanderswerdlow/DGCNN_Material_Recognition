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
from torch_geometric.utils import to_dense_batch

pre_transform = T.NormalizeScale()

transforms3d = {"crop": 0,
                "dropout": 0.2,
                "rotate": 0,
                "mirror": 0}
transforms3d = {}

# need to fix transforms (flip, crop, dropout)

train_dataset = GeoMat(get_dataset_dir(), True, None, pre_transform, transforms3d=transforms3d)
test_dataset = GeoMat(get_dataset_dir(), False, None, pre_transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)


def train():
    model.train()

    train_loss, train_pred, train_true = 0, [], []
    for data in tqdm(train_loader):
        optimizer.zero_grad()

        feat = torch.stack([x_ for x_ in data.hha])
        points = torch.stack([x_ for x_ in data.point_cloud_3d])
        data.pos = points
        data.x = feat

        out = model(data)
        loss = criterion(out.to(device), data.y.to(device))
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

            data.pos = points
            data.x = feat

            pred = model(data).max(dim=1)[1]
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
def extract_features(loader, loader_name, nlayer=16):
    model.eval()
    model.obtain_intermediate(nlayer)

    features = SaveFeatures(list(model.children())[nlayer])

    for _, data in enumerate(loader):
        feat = torch.stack([x_ for x_ in data.hha]).to(device)
        points = torch.stack([x_ for x_ in data.point_cloud_3d]).to(device)

        data.pos = points
        data.x = feat

        graph = model(data)
        feat = features.features

        feat = feat.x
        if feat.size(0) == graph.x.size(0) and graph.x.size(0) == graph.pos.size(0) and graph.pos.size(0) == len(graph.batch):
            graph.x = feat
            x, _ = to_dense_batch(graph.x, batch=graph.batch)
            pos, _ = to_dense_batch(graph.pos, batch=graph.batch)
            labels = graph.y
            for i in range(0, x.size(0)):
                x_i = x[i, :, :]
                pos_i = pos[i, :, :]
                y_i = labels[i].unsqueeze(0).reshape(-1,1)

                save_name = f"{get_data_dir()}/fusion/3d/{loader_name}/{data.dataset_idx[i]}.h5"
                save_h5_features(save_name, "points", pos_i.detach().cpu().numpy())
                save_h5_features(save_name, "features", x_i.detach().cpu().numpy())
                save_h5_features(save_name, "label", y_i[0].detach().cpu().numpy())


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #          0                     1   2   3              4                     5   6   7              8                     9   10  11             12                    13  14  15             16                     17  18  19       20      21   
    conf_3d = 'multigraphconv_9_16_0,b_0,r_0,pnv_max_0.05_0,multigraphconv_9_16_0,b_0,r_0,pnv_max_0.08_0,multigraphconv_9_32_0,b_0,r_0,pnv_max_0.12_0,multigraphconv_9_64_0,b_0,r_0,pnv_max_0.24_0,multigraphconv_9_128_0,b_0,r_0,gp_avg_0,d_0.2_0,f_19_cp_0'
    model = GraphNetwork(config=conf_3d, nfeat=3, multigpu=True)

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model_name = os.path.basename(__file__).rstrip(".py")
    run_training(model_name, train, test, model, optimizer, scheduler, total_epochs=3)
    print("done")
    extract_features(train_loader, 'train')
    extract_features(test_loader, 'test')
