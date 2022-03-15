# Graph Convolution on RGBD data for material recognition

## Structure
- DGCNN Networks are in `dgcnn/`
- Fusion networks are in `fusion/`
- `geomat.py` processes the dataset and facilitates feature extraction for some models due to torch_geometric limitations

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

The GeoMat dataset can be found [here](http://josephdegol.com/pages/MatRec_CVPR16.html). You must download all 19 zip files (one per class) and extract the folders to `data/geomat/raw`.

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

geomat_classification_v7: (DG-V1), 1205075 params
geomat_classification_v1: (DG-V2), 1279251 params
geomat_classification_v6: (DG-V3), 1525075 params, 196259539 params (backbone)
geomat_classification_v8: (DG-V4), 926675 params, 87585939 params (backbone)

To get number of params: `print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))`

