import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from geomat import GeoMat
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
import shutil
import os.path
from util import load_ckp, save_ckp, criterion

path = osp.join(osp.dirname(osp.realpath(__file__)), "data/geomat")
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)  # T.SamplePoints(1024)
train_dataset = GeoMat(path, True, transform, pre_transform)
test_dataset = GeoMat(path, False, transform, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=6)


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


def train():
    model.train()

    train_loss, train_pred, train_true = 0, [], []
    for data in train_loader:
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
last_checkpoint = f"data/checkpoints/{model_name}_best_model.pt"

if os.path.isfile(last_checkpoint):
    model, optimizer, start_epoch = load_ckp(last_checkpoint, model, optimizer, scheduler)
else:
    start_epoch = 1

writer = SummaryWriter()
best_test_acc = 0
for epoch in range(start_epoch, 201):
    loss, train_acc, balanced_train_acc = train()
    test_acc = test(test_loader)
    print(f"Epoch {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Balanced Train Acc: {balanced_train_acc:.4f}, Test: {test_acc:.4f}")
    scheduler.step()

    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Balanced_Accuracy/train", balanced_train_acc, epoch)
    writer.add_scalar("Accuracy/test", test_acc, epoch)

    checkpoint = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
    save_ckp(checkpoint, test_acc >= best_test_acc, "data/checkpoints", model_name)

    best_test_acc = max(test_acc, best_test_acc)

writer.close()
