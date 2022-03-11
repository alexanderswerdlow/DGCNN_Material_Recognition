import os
import os.path as osp
import scipy.io as sio
import torch
import numpy as np
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from PIL import Image
import torchvision
from torch_geometric.data import Data, Dataset
import fusion.convnext

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "image":
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


def create_point_cloud_depth(img, depth, fx, fy, cx, cy):
    depth_shape = depth.shape
    [x_d, y_d] = np.meshgrid(range(0, depth_shape[1]), range(0, depth_shape[0]))
    x3 = np.divide(np.multiply((x_d - cx), depth), fx)
    y3 = np.divide(np.multiply((y_d - cy), depth), fy)
    z3 = depth

    return np.stack((x3, y3, z3), axis=2), img.reshape(-1, img.shape[-1])


class GeoMat(Dataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, feature_extraction=None):

        self.train_raw = self.read_txt(osp.join(root, "raw_train.txt"))
        self.test_raw = self.read_txt(osp.join(root, "raw_test.txt"))
        self.train_proc = self.read_txt(osp.join(root, "processed_train.txt"))
        self.test_proc = self.read_txt(osp.join(root, "processed_test.txt"))

        super().__init__(root, transform, pre_transform, pre_filter)
        self.train = train
        self.data = self.train_proc if self.train else self.test_proc

        self.feature_extraction = feature_extraction

        if self.feature_extraction:
            self.rgb_shape = np.empty((100, 100)).shape
            self.boxes = np.zeros((self.rgb_shape[0], self.rgb_shape[1], 5))
            for i in range(self.rgb_shape[0]):
                for j in range(self.rgb_shape[1]):
                    self.boxes[i][j][1:] = np.array([i, j, i + 1, j + 1])

            self.boxes = torch.from_numpy(self.boxes.reshape(-1, 5)).float().cuda()
            if self.feature_extraction == 'v2':
                self.img_model = timm.create_model("efficientnet_b3a", features_only=True, pretrained=True).cuda()
            elif self.feature_extraction == 'v3':
                self.img_model = timm.create_model('convnext_base', pretrained=True).cuda()
            else:
                raise Exception('Invalid Feature Extraction Value')
            
            self.img_model.eval()
            self.img_config = resolve_data_config({}, model=self.img_model)
            self.img_transform = create_transform(**self.img_config)

    @property
    def raw_file_names(self):
        return self.train_raw + self.test_raw

    @property
    def processed_file_names(self):
        return self.train_proc + self.test_proc

    def read_txt(self, txt):
        with open(txt, "r") as f:
            return [fn.strip() for fn in f.readlines()]

    def len(self):
        return len(self.data)

    def get(self, idx):
        fn = self.data[idx]
        data = torch.load(osp.join(self.processed_dir, fn), map_location="cpu")
        if self.feature_extraction:
            _, _, _, img = data.pos, data.x, data.batch, data.image
            with torch.no_grad():
                img_batch = self.img_transform(Image.fromarray(img.numpy())).cuda()
            if self.feature_extraction == 'v2':
                unpooled_features = self.img_model(img_batch.unsqueeze(0))[-2]
            elif self.feature_extraction == 'v3':
                unpooled_features = self.img_model.get_features_concat(img_batch.unsqueeze(0))
            data.features = torchvision.ops.ps_roi_align(unpooled_features, self.boxes, 1).squeeze()
        return data

    def process(self):

        raw_filenames = self.raw_paths
        processed_filenames = self.processed_paths

        for dataset_idx, (raw_fn, proc_fn) in enumerate(zip(raw_filenames, processed_filenames)):
            f = sio.loadmat(raw_fn)

            label = torch.from_numpy(f["Class"][0]).to(torch.long) - 1  # Labeled 1-19 but pytorch functions expect zero-indexing so convert to 0-18
            depth = np.ascontiguousarray(f["Depth"].astype(np.float32))  # 100x100
            rgb = np.ascontiguousarray(f["Image"])  # 100x100x3
            intrinsics = f["Intrinsics"].astype(np.float64)  # 3x3
            extrinsics = np.vstack([f["Extrinsics"].astype(np.float64), [0, 0, 0, 1]])  # 3x4

            depth_, img_ = create_point_cloud_depth(rgb, -depth, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
            data = MyData(pos=torch.from_numpy(depth_.reshape(-1, 3)), x=torch.from_numpy(img_), y=label)
            data.image = torch.from_numpy(rgb)
            data.dataset_idx = dataset_idx
            data.img_point_cloud = depth_

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            os.makedirs(osp.dirname(proc_fn), exist_ok=True)
            torch.save(data, proc_fn)


if __name__ == "__main__":
    g = GeoMat("data/geomat")
