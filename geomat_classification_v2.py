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
from util import criterion, run_training
import os.path
from tqdm import tqdm

path = osp.join(osp.dirname(osp.realpath(__file__)), "data/geomat")
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)  # T.SamplePoints(1024)
train_dataset = GeoMat(path, True, transform, pre_transform, feature_extraction=True)
test_dataset = GeoMat(path, False, transform, pre_transform, feature_extraction=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=20, aggr="max"):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * (3 + 3 + 136), 64], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.fc1 = MLP([64 + 64 + 128 + 256, 1024], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=0.5)
        self.fc2 = MLP([1024, 512, 256, out_channels], dropout=0.5)

    def forward(self, data):
        pos, x, batch, features = data.pos, data.x, data.batch, data.features
        x1 = self.conv1(torch.cat((pos, x, features), dim=1).float(), batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        out = self.fc1(torch.cat((x1, x2, x3, x4), dim=1))
        out = global_max_pool(out, batch)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


def train():
    model.train()

    train_loss, train_pred, train_true = 0, [], []
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        preds = out.max(dim=1)[1]
        train_loss += loss.item() * data.num_graphs
        train_true.append(data.y.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return train_loss / len(train_dataset), metrics.accuracy_score(train_true, train_pred), metrics.balanced_accuracy_score(train_true, train_pred)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(in_channels=6, out_channels=19, k=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
model_name = os.path.basename(__file__).rstrip(".py")

run_training(model_name, train, test, model, optimizer, scheduler, total_epochs=200)
