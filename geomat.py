import os
import os.path as osp
import scipy.io as sio
import torch
import numpy as np
import open3d as o3d

from torch_geometric.data import (Data, Dataset)

class GeoMat(Dataset):

    def __init__(self, root, train=True, transform=None,
                pre_transform=None, pre_filter=None):

        self.train_raw = self.read_txt(osp.join(root, 'raw_train.txt'))
        self.test_raw = self.read_txt(osp.join(root, 'raw_test.txt'))
        self.train_proc = self.read_txt(osp.join(root, 'processed_train.txt'))
        self.test_proc = self.read_txt(osp.join(root, 'processed_test.txt'))

        super().__init__(root, transform, pre_transform, pre_filter)
        self.train = train
        self.data = self.train_proc if self.train else self.test_proc

    @property
    def raw_file_names(self):
        return self.train_raw + self.test_raw
        
    @property
    def processed_file_names(self):
        return self.train_proc + self.test_proc
    
    def read_txt(self, txt):
        with open(txt, 'r') as f:
            return [fn.strip() for fn in f.readlines()]

    def len(self):
        return len(self.data)

    def get(self, idx):
        fn = self.data[idx]
        return torch.load(osp.join(self.processed_dir, fn))
    
    def process(self):

        raw_filenames = self.raw_paths
        processed_filenames = self.processed_paths

        for raw_fn, proc_fn in zip(raw_filenames, processed_filenames):

            f = sio.loadmat(raw_fn)

            label = torch.from_numpy(f['Class'][0]).to(torch.long) - 1 # Labeled 1-19 but pytorch functions expect zero-indexing so convert to 0-18
            depth = np.ascontiguousarray(f['Depth'].astype(np.float32)) # 100x100
            rgb = np.ascontiguousarray(f['Image']) # 100x100x3
            intrinsics = f['Intrinsics'].astype(np.float64) # 3x3
            extrinsics = np.vstack([f['Extrinsics'].astype(np.float64), [0, 0, 0, 1]]) # 3x4

            # TODO: Since we invert depth here, we probably need to invert z coord in extrinsic...
            im_rgb, im_depth = o3d.geometry.Image(rgb), o3d.geometry.Image(-depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_rgb, im_depth, convert_rgb_to_intensity=False)

            # width, height, fx, fy, cx, cy
            intrinsics = o3d.camera.PinholeCameraIntrinsic(rgb.shape[1], rgb.shape[0], intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2])
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsics, project_valid_depth_only=True)

            # breakpoint()
            # o3d.visualization.draw_geometries([pcd]) # uncomment for viz

            # pcd.estimate_normals()
            # distances = pcd.compute_nearest_neighbor_distance()
            # avg_dist = np.mean(distances)
            # radius = 1.5 * avg_dist
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

            # pcd.points is n x 3, pcd.colors is n x 3 (I'm pretty sure RGB [0, 1])
            pointcloud = torch.from_numpy(np.asarray(pcd.points))
            pointcloud_rgb = torch.from_numpy(np.asarray(pcd.colors))
            # TODO: Add surface normals at some point ofc, we prob don't need intrinsics in xs, maybe later we'll generate mesh using open3d, that I can look into

            x = torch.cat((pointcloud, pointcloud_rgb), dim=1)
            y = label

            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            os.makedirs(osp.dirname(proc_fn), exist_ok=True)
            torch.save(data, proc_fn)
    

if __name__ == "__main__":
    g = GeoMat('data/geomat')