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
* os
* json
* numpy
* pandas
* matplotlib
* cv2
* torch 
* torchvision
* CUDA 10.1
* pycocotools

```!pip install pyyaml==5.1```

```!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html```

## Step

a.Set up the requirement 

b.Preprocessing

c.detectron2

## Snapshot of google drive
![](https://i.imgur.com/V3auFDQ.png)

## Reference
* cocodataset/cocoapi : https://github.com/cocodataset/cocoapi/tree/master/PythonAPI

* NCTU-VRDL/CS_IOC5008 : https://github.com/NCTU-VRDL/CS_IOC5008/blob/master/HW4/data_loader.ipynb

* facebookresearch/detectron2 : https://github.com/facebookresearch/detectron2?fbclid=IwAR1x0mEokXtgA8St7FnyEKzeX-Ok8rj-aH_Fn7SqSY_nhyxGdaRz3OjgTCk
