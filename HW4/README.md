# IOC5008_HW4

## Introduction

* Input
    * Training set : 291 high-resolution images
    * Testing set: 14 low-resolution images

* Output
    * 14 high-resolution images
    * Need to upscale the test images with an upscaling factor of 3 (360x360)

* Baseline
    * PSNR : 25.03 
   
* Goal
    * Train a model to reconstruct a high-resolution image from a low-resolution input 
    * There are no annotations in the image super-resolution task.
    * Create the HR-LR image pairs by the provided HR images.
    * DO NOT use any pre-trained models or external data.
    
* Evaluation metrics
    * Peak signal to noise ratio (PSNR) measures the similarity between two images.
    
## Hardware
* Intel(R) Core(TM) i7-9700K CPU @3.6GHz
* NVIDIA GeForce RTX2060

下面還沒改
## Requirement
- Numpy
- PIL
- gdal
- scipy
- scikit-learn
- opencv-python
- tqdm
- jupyter
- ipywidgets
- IPython
- torch
- torchvision
- CUDA 10.1

## Step

a.Set up the requirement 

b.Run ``` create pair.py```

c.Run Model - VDSR

## Reference
* Create HR-LR image pairs github
   * https://github.com/Paper99/SRFBN_CVPR19
   * https://github.com/jshermeyer/RFSR
   * https://github.com/S-aiueo32/srntt-pytorch
* Model
   * ImageSuper-Resolution (paper with codes)
   https://paperswithcode.com/task/image-super-resolution
   * SRGAN
   https://arxiv.org/pdf/1609.04802.pdf
   * RFSR
   https://github.com/jshermeyer/RFSR
   * VDSR
   https://github.com/twtygqyy/pytorch-vdsr
   https://cv.snu.ac.kr/research/VDSR/
   * BasicSR (Basic Super Restoration)
   https://github.com/xinntao/BasicSR
