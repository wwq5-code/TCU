# TCU-S



# Triplet Contrastive Unlearning with Self Mode Connectivity for Manifold Representation-based Unlearning

## Overview
This repository is the official implementation of TCU-S, and the corresponding paper is under review.


## Prerequisites

```
python = 3.10.10
torch==2.0.0
torchvision==0.15.1
matplotlib==3.7.1
numpy==1.23.5
```

We also show the requirements packages in requirements.txt


## Artifact Evaluation

Here, we demonstrate the overall evaluations, which are also the main achievement claimed in the paper. We will explain the results and demonstrate how to achieve these results using the script and corresponding parameters.

Evaluated on NVIDIA Quadro RTX 6000 GPUs,

### TABLE I: Performance overview of various machine unlearning methods with MMCRs:

On MNIST, USS = 200

| On MNIST | MIA (%)      | RA (%)   |   TA (%) |  RT (second)  |
| --------  | --------    | -------- | -------- | -------- |  
| Retraining | 63.00      | 99.49   |  99.23  | 470.4  | 
| GA        | 64.50      | 99.01   | 98.81   | 0.202  |  
| VBU       | 57.00       | 99.37       | 99.15     | 0.201     | 
| RFU       | 53.50    | 99.39    | 99.28      | 0.354      |  
| SalUn       | 55.00    | 99.37    | 99.22      | 1.839      |  
| TCU-S (Our)  | 61.50      | 99.57    | 99.14   | 1.483     |  

In this table, we can achieve these metric values by running corresponding python files.

1. To run the TCU on MNIST, we can run
```
python /Manifold_unl/On_MNIST/26Mar25/MNIST_Normal_distribution.py
```

2. To run the TCU on CIFAR10, we can run
```
python /Manifold_unl/On_CIFAR10/26Mar2025/CIFAR10_Normal_distribution.py
```

3. To run the TCU on CelebA, we can run
```
python /Manifold_unl/On_CelebA/18Apr2025/CELEBA_Normal_distribution_new.py
```

Note that, to sucessfully run the program on CelebA, we need first prepare the CelebA dataset, which can be downloaded from: 
(https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)
 

4. To run the TCU on Tiny-ImageNet, we can run
```
python /Manifold_unl/On_tiny_IMAGENET/IMAGEnet_Normal_distribution_new.py
```
The Tiny-ImageNet dataset can be downloaded from: (https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet/data)
