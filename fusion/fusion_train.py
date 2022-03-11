import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from geomat import GeoMat
from torch_geometric.loader import DataLoader
import sklearn.metrics as metrics
import os.path
from util import criterion, run_training, get_data_dir
from tqdm import tqdm
from fusion.networks import TwoStreamNetwork
from fusion.fusion_dataset import FusionDataset
from fusion.radam import RAdam

path = osp.join(osp.dirname(osp.realpath(__file__)), "data/geomat")

train_dataset = FusionDataset('train', f'{get_data_dir()}/fusion/3d', f'{get_data_dir()}/fusion/2d')
test_dataset = FusionDataset('test', f'{get_data_dir()}/fusion/3d', f'{get_data_dir()}/fusion/2d')

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=6)

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
model = TwoStreamNetwork('b,r,gp_avg,d_0.5,f_19_cp_1', features_b1=128, features_b2=896, rad_fuse_pool=0.24, features_proj_b1=256, features_proj_b2=256, proj_b1=True, proj_b2=True).to(device)
optimizer = torch.optim.RAdam(model.parameters(), betas=(0.9, 0.999), lr=0.001, wd=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
model_name = os.path.basename(__file__).rstrip(".py")

run_training(model_name, train, test, model, optimizer, scheduler, total_epochs=200)

