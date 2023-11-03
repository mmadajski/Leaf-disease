# Leaf Infection Segmentation with U-Net :leaves:

## Overview

This project implements a U-Net neural network to segment infected parts of a leaf from an input image. The model is trained using Python and TensorFlow, and it is designed for applications in plant disease detection and agriculture.

## Data 

The data is contained in two directories containing images and their masks.
Since the images were different sizes, they were all resized to 256x256, as were their masks.
The images were normalized to the [0, 1] range by dividing by 255.
The data is then divided into a training set and a test set (70% of the observations in the training set).

## Network architecture

Implemented neural network is standard U-Net architecture. 
The network takes as input a normalized image of size [256, 256, 3].
The input image passes through three blocks, each block contains two convolution layers with Relu activation function, followed by a 2x2 MaxPool layer (except for the last layer).
Convolution layers have 16, 32 and 64 filters, respectively.

The expansive path has two similar blocks, although they have a Conv2DTranspose layer, followed by a concentration with skip connections at the beginning. 
Conv2DTranspose layers have a kernel size [2, 2].
Convolutional layers in these blocks have 32 and 16 filters respectively. 

Finally, there is another convolution layer, which has 1 filter and a sigmoidal activation function. 

## Training 

The model is trained using the adam optimizer, binary cross entropy as a loss function, 15 epochs and batch size equal to 25.

## Results

The model archived IOU equal to 0.48 on training dataset and 0.46 on test dataset.
The low value of the IOU metric suggests that the model is performing poorly. This is likely due to a large imbalance in the target class.

Sample image: 

![alt](https://github.com/mmadajski/Leaf-disease/blob/main/examples/Sample_8.png?raw=true)


## Installation

1. Download the requirements and source code from the latest release and unzip.
2. Create an environment for your project and install the requirements. 
If you are using, pip you can use the following command:
```bash
pip install -r requirements.txt
```
3. Download and unzip in your project the data from: [Kaggle](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset/data).
4. You should be ready to go, just run U-net.py :smiley:


