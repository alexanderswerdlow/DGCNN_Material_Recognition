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
from util import criterion, run_training, get_data_dir
import os.path
from tqdm import tqdm
import timm
import fusion.convnext

path = osp.join(osp.dirname(osp.realpath(__file__)), "data/geomat")
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
img_model = timm.create_model("convnext_large", num_classes=19, drop_path_rate=0.8).cuda()
checkpoint = torch.load(f"{get_data_dir()}/checkpoints/texture_train_large_best_model.pt")
img_model.load_state_dict(checkpoint["state_dict"])
del checkpoint
train_dataset = GeoMat(path, True, transform, pre_transform, feature_extraction="v4", img_model=img_model)
test_dataset = GeoMat(path, False, transform, pre_transform, feature_extraction="v4", img_model=img_model)
train_loader = DataLoader(train_dataset, batch_size=19, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=19, shuffle=False, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=20, aggr="max", feat_size=2880):
        super().__init__()
        self.filter_conv = nn.Conv2d(feat_size, 32, 1)  # reduce filter size
        self.conv1 = DynamicEdgeConv(MLP([2 * (3 + 3 + 32), 64], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=0.5), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=0.5), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=0.5), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=0.5), k, aggr)
        self.fc1 = MLP([64 + 64 + 128 + 256, 1024], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=0.5)
        self.fc2 = MLP([1024, 512, 256, out_channels], dropout=0.5)

    def forward(self, data):
        # Feature Extraction must be in geomat.py so that torch_geometric can properly sample points
        pos, x, batch, features = data.pos, data.x, data.batch, data.features
        features = self.filter_conv(torch.unsqueeze(torch.unsqueeze(features, dim=-1), dim=-1)).squeeze()
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
    return (
        train_loss / len(train_dataset),
        metrics.accuracy_score(train_true, train_pred),
        metrics.balanced_accuracy_score(train_true, train_pred),
    )


def test():
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(in_channels=6, out_channels=19, k=40).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
model_name = os.path.basename(__file__).rstrip(".py")

run_training(model_name, train, test, model, optimizer, scheduler, total_epochs=200)
