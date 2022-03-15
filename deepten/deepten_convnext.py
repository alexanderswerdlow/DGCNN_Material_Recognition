import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from geomat import GeoMat
from torch_geometric.loader import DataLoader
import sklearn.metrics as metrics
import os.path
from util import criterion, run_training, get_data_dir, ConfusionMatrixMeter
from tqdm import tqdm
import encoding
from encoding.nn import Encoding, View, Normalize
import timm
import fusion.convnext

img_model = timm.create_model(
    "convnext_large", num_classes=19, drop_path_rate=0.8
).cuda()
checkpoint = torch.load(
    f"{get_data_dir()}/checkpoints/texture_train_large_best_model.pt"
)
img_model.load_state_dict(checkpoint["state_dict"])
del checkpoint

path = osp.join(osp.dirname(osp.realpath(__file__)), "data/geomat")
pre_transform = T.NormalizeScale()
train_dataset = GeoMat(path, True, pre_transform)
test_dataset = GeoMat(path, False, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)


class DeepTen(nn.Module):
    def __init__(self, nclass, pretrained):
        super(DeepTen, self).__init__()
        self.pretrained = pretrained
        n_codes = 32
        self.head = nn.Sequential(
            nn.Conv2d(1024, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128, K=n_codes),
            View(-1, 128 * n_codes),
            Normalize(),
            nn.Linear(128 * n_codes, nclass),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pretrained.forward_features(x)
        return self.head(x)


def train():
    model.train()
    cm = ConfusionMatrixMeter(cmap="Oranges")
    train_loss, train_pred, train_true = 0, [], []
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.image.permute(0, 3, 1, 2).float())
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        preds = out.max(dim=1)[1]
        train_loss += loss.item() * data.num_graphs
        train_true.append(data.y.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        cm.add(data.y.cpu().numpy(), preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return (
        train_loss / len(train_dataset),
        metrics.accuracy_score(train_true, train_pred),
        metrics.balanced_accuracy_score(train_true, train_pred),
        cm,
    )


def test():
    model.eval()
    correct = 0
    cm = ConfusionMatrixMeter(cmap="Blues")
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.image.permute(0, 3, 1, 2).float()).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        cm.add(data.y.detach().cpu().numpy(), pred.detach().cpu().numpy())
    return correct / len(test_loader.dataset), cm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepTen(19, img_model).to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
model_name = os.path.basename(__file__).rstrip(".py")

run_training(
    model_name, train, test, model, optimizer, scheduler, total_epochs=200, cm=True
)
