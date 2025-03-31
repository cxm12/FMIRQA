# Method
"Assessing AI-Reconstructed Hallucinations in Fluorescence Microscopy Image"

### Requirements
* Python 3.7
* CUDA 11.4 and CUDNN 
* Packages: 
  
  basicsr          ==          1.4.2
  easydict         ==          1.11.dev0
  imageio          ==          2.13.3
  keras            ==          2.11.0
  numpy            ==          1.21.5
  opencv-python    ==          4.5.4.60
  Pillow           ==          9.0.1
  scikit-image     ==          0.19.2
  scipy            ==          1.7.3
  tensorflow-gpu   ==          2.7.0
  tifffile         ==          2021.11.2
  torch            ==          1.10.0+cu113
  
csbdeep [![PyPI version](https://badge.fury.io/py/csbdeep.svg)](https://pypi.org/project/csbdeep)


### Prediction
The proposed HallAssess uses the well-trained models (UniFMIR [1]) and does not need additional training.

Evaluating the quality of super-resolution results
Download the pretrained models from `https://pan.baidu.com/s/1_NBMYfPrMylp71DPJFtcJg?pwd=ohe5`. Place the file in the model file. 

```
cd <directory of the .py file>
python mainUni2IQA.py
```
Replacing "reconstructpath" with the folder name containing AI-reconstructed images (such as super-resolved images), which need to be measured. 

Replacing "inputpath" with the folder name containing low-quality images (such as low-resolution images), which are the input of the AI models (such as super-resolution models) for generating AI-reconstructed images. 

### Data
All training and test data involved in the experiments are publicly available datasets. 

* The 3D denoising/isotropic reconstruction/projection datasets can be downloaded from `https://publications.mpi-cbg.de/publications-sites/7207/`

* The SR dataset can be downloaded from `https://doi.org/10.6084/m9.figshare.13264793`

### Model
The pretrained models can be downloaded from `https://pan.baidu.com/s/1_NBMYfPrMylp71DPJFtcJg?pwd=ohe5`


[1] Ma, C., Tan, W., He, R. et al. Pretraining a foundation model for generalizable fluorescence microscopy-based image restoration. Nat Methods 21, 1558–1567 (2024).
