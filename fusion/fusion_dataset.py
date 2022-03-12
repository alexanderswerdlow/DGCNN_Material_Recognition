"""
    2D-3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
from torch_geometric.data import Data
import math
import random
import numpy as np
import h5py
from torch_geometric.data import Batch
from os import walk


def custom_collate(data_list):
    batch_1 = Batch.from_data_list([d[0] for d in data_list])
    batch_2 = Batch.from_data_list([d[1] for d in data_list])
    return batch_1, batch_2


def dropout(P, F, p):
    idx = random.sample(range(P.shape[0]), int(math.ceil((1 - p) * P.shape[0])))
    return P[idx, :], F[idx, :] if F is not None else None


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, train_or_test, h5_folder_b1, h5_folder_b2):
        self.geometric_feat_folder = f"{h5_folder_b1}/{train_or_test}"
        self.texture_feat_folder = f"{h5_folder_b2}/{train_or_test}"
        self.files = sorted(next(walk(self.texture_feat_folder), (None, None, []))[2])

    def __getitem__(self, index):
        h5_file_b1 = h5py.File(f"{self.geometric_feat_folder}/{self.files[index]}", "r")
        h5_file_b2 = h5py.File(f"{self.texture_feat_folder}/{self.files[index]}", "r")

        cls_b1 = int(np.asarray((h5_file_b1["label"])))
        cls_b2 = int(np.asarray((h5_file_b2["label"])))

        if cls_b1 != cls_b2:
            raise RuntimeError("Branches have different classes")

        P_b1 = np.asarray(h5_file_b1["points"])
        P_b2 = np.asarray(h5_file_b2["points"])

        F_b1 = np.asarray(h5_file_b1["features"], dtype=np.float32)
        if len(F_b1.shape) == 1:
            F_b1 = np.transpose([F_b1])

        F_b2 = np.asarray(h5_file_b2["features"], dtype=np.float32)
        if len(F_b2.shape) == 1:
            F_b2 = np.transpose([F_b2])

        P_b1 -= np.min(P_b1, axis=0)
        P_b2 -= np.min(P_b2, axis=0)

        # print(F_b1.shape, P_b1.shape, cls_b1, F_b2.shape, P_b2.shape, cls_b2)
        data_1 = Data(x=torch.tensor(F_b1), pos=torch.tensor(P_b1), y=cls_b1)
        data_2 = Data(x=torch.tensor(F_b2), pos=torch.tensor(P_b2), y=cls_b2)

        return data_1, data_2

    def __len__(self):
        return len(self.files)
