import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from geomat import GeoMat
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
import sklearn.metrics as metrics
from util import run_training, get_data_dir, ConfusionMatrixMeter
import os.path
from tqdm import tqdm
import timm
from timm.loss import LabelSmoothingCrossEntropy

path = f"{get_data_dir()}/geomat"
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1000)
train_dataset = GeoMat(path, True, transform, pre_transform)
test_dataset = GeoMat(path, False, transform, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr="max"):
        super().__init__()
        self.conv1 = DynamicEdgeConv(
            MLP(
                [2 * 3, 64],
                act="LeakyReLU",
                act_kwargs={"negative_slope": 0.2},
                dropout=0.8,
            ),
            k,
            aggr,
        )
        self.conv2 = DynamicEdgeConv(
            MLP(
                [2 * 64, 128],
                act="LeakyReLU",
                act_kwargs={"negative_slope": 0.2},
                dropout=0.8,
            ),
            k,
            aggr,
        )
        self.conv3 = DynamicEdgeConv(
            MLP(
                [2 * 128, 256],
                act="LeakyReLU",
                act_kwargs={"negative_slope": 0.2},
                dropout=0.8,
            ),
            k,
            aggr,
        )
        self.fc1 = MLP(
            [256 + 128 + 64, 1024],
            act="LeakyReLU",
            act_kwargs={"negative_slope": 0.2},
            dropout=0.8,
        )
        self.fc2 = MLP([1024, 512, 256, out_channels], dropout=0.8)

    def forward(self, data):
        pos, batch = data.pos.cuda(), data.batch.cuda()
        x1 = self.conv1(pos.float(), batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.fc1(torch.cat((x1, x2, x3), dim=1))
        out = global_max_pool(out, batch)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


def train():
    model.train()
    cm = ConfusionMatrixMeter(cmap="Oranges")
    train_loss, train_pred, train_true = 0, [], []
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        out = model.forward(data)
        loss = criterion(out, data.y.cuda())
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
    cm = ConfusionMatrixMeter(cmap="Blues")
    correct = 0
    for data in test_loader:
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y.cuda()).sum().item()
        cm.add(data.y.detach().cpu().numpy(), pred.detach().cpu().numpy())

    return correct / len(test_loader.dataset), cm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(out_channels=19, k=40).to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
model_name = os.path.basename(__file__).rstrip(".py")

run_training(
    model_name, train, test, model, optimizer, scheduler, total_epochs=200, cm=True
)
