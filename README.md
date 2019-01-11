### Introduction

This is refactor of the PointNet++ semantic segmentation implementation; the main difference is use of the Tensorflow Data API. The original code is on the author's repo <a href="https://github.com/charlesq34/pointnet2">here</a> and the paper is <a href="https://arxiv.org/abs/1706.02413">here</a>. There you can find more information on PointNet.

My primary motivations are to process my own point cloud data, improve speed, and add documentation. At the moment I'm focusing only on semantic segmentation.

#### Todo list:
- [x] Cleanup semantic segmentation implementation
- [ ] Use Data API for training pipeline
- [x] Add more documentation
- [ ] Enable multi-gpu 


### Installation

I'm using Ubuntu 18.04, a Python 2 virtual environment, and the custmized TF operators. It's highly recommended that you have access to a GPU (Find installation support <a href="https://www.tensorflow.org/install/gpu">here</a>)

#### Setup virtualenv

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        
Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

### Usage

#### Downloading data

The paper uses pickled data derived from the ScanNet dataset. The authors have preprocessed data for you to download <a href="https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip">here (1.72GB)</a>. My code expects to find data here: `pointnet_datasets/data/scannet_data_pointnet2/*.pickle`.

```
cd pointnet_datasets
mkdir -p data/scannet_data_pointnet2 && cd data/scannet_data_pointnet2
wget https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip
unzip scannet_data_pointnet2.zip
```

#### Training/Testing Model

The master branch should run with the following command:

`python train.py`
