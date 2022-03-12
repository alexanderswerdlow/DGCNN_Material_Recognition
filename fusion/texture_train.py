import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import sklearn.metrics as metrics
import os.path
from util import run_training, get_dataset_dir, get_data_dir, load_ckp, save_h5_features
import torchvision.transforms as transforms
from geomat import GeoMat
from tqdm import tqdm
import timm
from timm.loss import LabelSmoothingCrossEntropy

pre_transform = T.NormalizeScale()
train_dataset = GeoMat(get_dataset_dir(), True, None, pre_transform)
test_dataset = GeoMat(get_dataset_dir(), False, None, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
val_transform = transforms.Compose([transforms.ToTensor(), normalize])


def train():
    model.train()

    train_loss, train_pred, train_true = 0, [], []
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        input = torch.stack(([train_transform(transforms.ToPILImage()(x_.permute(2, 0, 1))) for x_ in data.image])).to(device)
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
            input = torch.stack(([val_transform(transforms.ToPILImage()(x_.permute(2, 0, 1))) for x_ in data.image])).to(device)
            pred = model(input).max(dim=1)[1]
        correct += pred.eq(data.y.to(device)).sum().item()
    return correct / len(test_loader.dataset)

@torch.no_grad()
def extract_features(loader, loader_name):
    for _, data in enumerate(loader):
        input = torch.stack(([val_transform(transforms.ToPILImage()(x_.permute(2, 0, 1))) for x_ in data.image])).to(device)
        features = model.get_features_concat_pool_only(input)
        for i in range(0, features.shape[0]):
            feature = features[i].view(features.size(1), -1).permute(1, 0)

            # We have 100x100 3d points w/896 filters each size 14x14 so we need to pool to go from 100x100 -> 14x14
            points_average = torch.nn.AvgPool2d(7, count_include_pad=False, ceil_mode=False)
            points_torch_32 = points_average(torch.tensor(data.img_point_cloud[i]).permute(2, 0, 1))
            points_d32 = np.reshape(points_torch_32.data.numpy(), (-1, 3))

            save_name = f"{get_data_dir()}/fusion/2d/{loader_name}/{data.dataset_idx[i]}.h5"
            save_h5_features(save_name, "features", feature.detach().cpu().numpy())
            save_h5_features(save_name, "points", points_d32)
            save_h5_features(save_name, "label", np.atleast_1d(data.y[i].detach().cpu().numpy()))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("convnext_base", pretrained=True, num_classes=19, drop_path_rate=0.8).cuda()
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    model_name = os.path.basename(__file__).rstrip(".py") + "_base"

    run_training(model_name, train, test, model, optimizer, scheduler, total_epochs=50)

    model, optimizer, start_epoch = load_ckp("data/checkpoints/texture_train_base_best_model.pt", model, optimizer, scheduler)
    extract_features(train_loader, "train")
    extract_features(test_loader, "test")