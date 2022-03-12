# Dynamic Graph Convolution (DGCNN) on RGBD data for material recognition


## Install instructions

*Note: Tested on Ubuntu 20.04 w/CUDA 11.3 and Python 3.8.12. If using another CUDA version/OS please see the [PyTorch](https://pytorch.org/get-started/locally/) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) install docs and replace the torch\* entries in `requirements.txt`*

```shell
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

To install torch-geometric you may need the following env variables to include cuda as follows:
```
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Run instructions

```shell
tensorboard --logdir=runs

python -m geomat_classification_v1
python -m geomat_classification_v2
python -m geomat_classification_v3
python -m geomat_classification_v4
python -m deepten

python -m fusion.texture_train
python -m fusion.geometric_train
python -m fusion.fusion_train
```

To clear data from a model:

```
rm -rf runs/geomat_classification_v3 && rm data/checkpoints/geomat_classification_v3*
```

To clear features:

```
rm -f data/fusion/2d/test/* && rm -f data/fusion/2d/train/*
```

## Training log

geomat_classification_v1: No image features used, knn=20, lr=0.001, Adam
geomat_classification_v2: Used efficientnet_b3a pretrained features (136 dim) from second to last layer, knn=40, lr=0.001, Adam
geomat_classification_v3: Used convnext_base pretrained features (all 4 layers to 14x14x1920), knn=40, lr=0.001, RAdam
geomat_classification_v4: Used convnext_base trained features from texture_train_large (all 4 layers to 14x14x1920), knn=40, lr=0.001, RAdam

texture_train: Used convnext_base, all 4 stages frozen, dropout=0
texture_train_large: Used convnext_large, stages 1 and prior params frozen, dropout=0.8