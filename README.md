# Dynamic Graph Convolution (DGCNN) on RGBD data for material recognition


## Install instructions

*Note: Tested on Ubuntu 20.04 w/CUDA 11.3 and Python 3.8.12. If using another CUDA version/OS please see the [PyTorch](https://pytorch.org/get-started/locally/) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) install docs and replace the torch\* entries in `requirements.txt`*

```shell
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Run instructions

```shell
tensorboard --logdir=runs
```