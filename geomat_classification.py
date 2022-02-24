import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from geomat import GeoMat
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

path = osp.join(osp.dirname(osp.realpath(__file__)), "data/geomat")
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)  # T.SamplePoints(1024)
train_dataset = GeoMat(path, True, transform, pre_transform)
test_dataset = GeoMat(path, False, transform, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

def criterion(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=20, aggr="max"):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * in_channels, 64], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}), k, aggr)
        self.fc1 = MLP([64 + 64 + 128 + 256, 1024], act="LeakyReLU", act_kwargs={"negative_slope": 0.2}, dropout=0.5)
        self.fc2 = MLP([1024, 512, 256, out_channels], dropout=0.5)

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        x1 = self.conv1(torch.cat((pos, x), dim=1).float(), batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        out = self.fc1(torch.cat((x1, x2, x3, x4), dim=1))
        out = global_max_pool(out, batch)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(in_channels=6, out_channels=19, k=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 201):
    loss = train()
    test_acc = test(test_loader)
    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}")
    scheduler.step()
