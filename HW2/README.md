# IOC5008_HW2
## Introduction

* Input
    * The Street View House Numbers (SVHN Dataset)
    * Trian : 33,402 images
    * Test : images 13,068 images Output

* Output
    * json file which has { bbox(y1,x1,y2,x2), probability, and label } Baseline


* Baseline
    * mAP : 0.36898
    * Speed : 558 ms per image 
    
* Goal
    * Train a digit detector (Accurate and Fast)

## Hardware
* Intel(R) Core(TM) i7-9700K CPU @3.6GHz
* NVIDIA GeForce RTX2060

## Requirement
torch 
torchvision
numpy
pandas
pathlib
CUDA 10.1

## Dataset
* train.zip
* test.zip

`unzip -uq "/content/gdrive/My Drive/train.zip" -d "/content/gdrive/My Drive/Dataset"`
* Annotation

use `SVHN_h5py.ipynb`to convert mat file to image
## Steps
a.Clone the DarkNet

b.Compile DarkNet using Nvidia GPU

`!sed -i 's/OPENCV=0/OPENCV=1/' Makefile`

`!sed -i 's/GPU=0/GPU=1/' Makefile`

`!sed -i 's/CUDNN=0/CUDNN=1/' Makefile`

c.Setting

* Dictionary the path : the training, train.txt
* Batch size : 80
* Learning rate :10E-5
* Use the Yolov3 pre-trained model

`cfg/yolov3.cfg`

`cfg/yolov3_training.cfg`

`cfg darknet53.conv.74`

the classes number to 10 classes, resize to 64X64

## Model Implementation
* mmlab(mmdetection/mmcv)
 https://github.com/open-mmlab/mmdetection
* DarkNet
https://github.com/AlexeyAB/darknet



