# Dynamic Graph Convolution (DGCNN) on RGBD data for material recognition


## Install instructions

*Note: only for Ubuntu 20.04 w/CUDA 11.3. If using another CUDA version/OS please see the [PyTorch](https://pytorch.org/get-started/locally/) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) install docs.*

```shell
pip install jupyter numpy scipy matplotlib scikit-image opencv-python pillow wheel cython h5py tqdm
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```