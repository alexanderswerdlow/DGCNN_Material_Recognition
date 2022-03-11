import torch
from torch_geometric.data import Data
import math
import random
import numpy as np
import transforms3d
import os
import h5py
import torch_cluster


def vertical_rot(M):
    angle = random.uniform(0, 2*math.pi)
    Mnew = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)
    return Mnew

def dropout(P, F, p):
    idx = random.sample(range(P.shape[0]), int(math.ceil((1-p)*P.shape[0])))
    return P[idx, :], F[idx, :] if F is not None else None

def mirror(r, M):
    if random.random() < r/2:
        Mnew = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
    if random.random() < r/2:
        Mnew = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), Mnew)
    return Mnew

def random_crop_3D(P, F, factor):
    npoints = P.shape[0]
    n_points_after_crop = np.round(npoints*factor).astype(np.int)

    points_max = (P.max(axis=0)*1000).astype(np.int)
    points_min = (P.min(axis=0)*1000).astype(np.int)

    centroid = np.asarray([np.random.randint(low=points_min[0], high=points_max[0], dtype=int),
                           np.random.randint(low=points_min[1], high=points_max[1], dtype=int),
                           np.random.randint(low=points_min[2], high=points_max[2], dtype=int)])

    centroid = centroid.astype(np.float32)/1000

    rad = 0.1
    inc = 0.2

    npoints_inside_sphere = 0

    x = torch.from_numpy(P)
    y = torch.from_numpy(centroid).unsqueeze(0)
    while npoints_inside_sphere < n_points_after_crop:
        _, crop = torch_cluster.radius(x, y, rad, max_num_neighbors=n_points_after_crop)

        npoints_inside_sphere = len(crop)

        rad = np.round(rad + inc, 1)

    return P[crop.numpy()], F[crop.numpy()]