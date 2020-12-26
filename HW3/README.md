# IOC5008_HW3

## Introduction

* Input
    * Tiny PASCAL VOC 
    * Trian : 1,349 images
    * Test : 100 images 
    * NO external data should be used but ImageNet
    * Deal with the overfitting problem

* Output
    * submission.json (coco style format)

* Baseline
    * mAP@0.5: 0.247 
   
* Goal
    * Instance segmentation

## Hardware
* Intel(R) Core(TM) i7-9700K CPU @3.6GHz
* NVIDIA GeForce RTX2060

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






