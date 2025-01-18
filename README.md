# TCU



# Triplet Contrastive Unlearning for Manifold Representation-based Unlearning

## Overview
This repository is the official implementation of TCU, and the corresponding paper is under review.


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

On MNIST, USR = 1%

| On MNIST | MIA (%)      | RA (%)   |   TA (%) |  RT (second)  |
| --------  | --------    | -------- | -------- | -------- |  
| Retraining | 56.00      | 99.34   |  99.06  | 469.95  | 
| GA        | 60.00      | 99.11   | 98.87   | 0.105  |  
| VBU       | 56.99       | 99.30       | 99.05     | 0.184     | 
| RFU       | 49.00    | 99.38    | 99.21      | 0.242      |  
| TCU (Our)  | 55.99      | 99.40    | 99.01   | 0.161     |  

In this table, we can achieve these metric values by running corresponding python files.

1. To run the TCU on MNIST, we can run
```
python /Manifold_unl/On_MNIST/New_version/MNIST_Normal_distribution.py
```

2. To run the TCU on CIFAR10, we can run
```
python /Manifold_unl/On_CIFAR10/New_version/CIFAR10_Normal_distribution.py
```

3. To run the TCU on CelebA, we can run
```
python /Manifold_unl/On_CelebA/New_version/CELEBA_Normal_distribution_new.py
```

Note that, to sucessfully run the program on CelebA, we need first prepare the CelebA dataset, which can be downloaded from: 
(https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)
 

