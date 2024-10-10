# DDM-Net

Official Pytorch Code base for "Complex agricultural parcel boundary extraction through semantic segmentation network with dual-task constraints"

[Project](https://github.com/hexianjia/DDM-Net-torch)

## Introduction

In this paper, we propose a semantic segmentation network named DDM-Net with three techniques to handle these issues. First, a dual-feature attention (DFA) module is employed between the encoder and decoder to extract category and multi-scale information.  It reduces the detail feature loss caused by multi-layer convolution and pooling to improve the recognition of blurred boundaries. Second, a new multi-feature extraction and fusion (MEF) module is proposed. It reduces the influence of irrelevant information for effective adaptation to complex agricultural parcel boundaries. Third, a dual-task network is designed, with the primary and auxiliary tasks devoting to boundary and mask mappings, respectively. This pairwise task provides dual perspectives to comprehensively capture complex agricultural parcels features, improving boundary continuity.

<p align="center">
  <img src="DDM-Net.png" width="800"/>
</p>

<p align="center">
  <img src="MEF.png" width="800"/>
</p>

<p align="center">
  <img src="result.png" width="800"/>
</p>



## Using the code:

The code is stable while using Python 3.7.0, CUDA >=11.0

- Clone this repository:
```bash
git clone https://github.com/hexianjia/DDM-Net-torch.git
cd DDM-Net-torch
```

To install all the dependencies using conda or pip:

```
PyTorch
TensorboardX
torchvision
OpenCV
numpy
tqdm
gdal
Imag
```

## Data Format

Make sure to put the files as the following structure:

```
inputs
└── <train>
    ├── image
    |   ├── 001.tif
    │   ├── 002.tif
    │   ├── 003.tif
    │   ├── ...
    |
    └── mask
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── boundary
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...

```

For test and validation datasets, the same structure as the above.

## Training and testing

1. Train the model.
```
python train.py --train_path ./fields/image --save_path ./model --model_type 
```
2. Evaluate.
```
python predict.py --model_file ./model/150.pt --save_path ./save  --val_path ./test_image
```

If you have any questions, you can contact us: kinghexianjia@gmail.com and yangmei@swpu.edu.cn.

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from DD-Net

