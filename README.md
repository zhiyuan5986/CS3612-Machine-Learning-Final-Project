# CS3612-01 Final Project

## Overview
In this project, I finished the Mandatory Task: Fashion-MNIST clothing classification with a Neural Network built by myself, and finished Optional task 2: Visualization methods to explain DNNs (for simplicity, I call this task as task 3). 

## Task1: Fashion-MNIST clothing classification
For task 1, you can see all my codes in directory `Task1`. 

### Run
You can run the codes starting from root directory:
```bash
cd Task1
python Task1.py --train
python Task1,py
```
Note that the first `python Task1.py --train` command set the code in training mode, and `python Task1.py` use the parameters trained just now to do testing with PCA and t-SNE.

What's more, I directly download the dataset using torchvision.datasets.FashionMNIST, and I also implement grayscale and resize image to (32,32).
### Files
```
- Task1
 |- Task1.py        # main script of task 1
 |- utils.py        # inplementation of PCA and t-SNE, and some visualization methods
 |- checkpoints     # (After you run training mode, this directory will be automatically built)
    |- v1.pt        # (The parameter of trained model)
 |- dataset         # (After you run training mode, this directory will be automatically built, which contains dataset)
 |- output          # (After you run training mode, this directory will be automatically built, which contains visualization results)
```

## Task3: Visualization methods to explain DNNs. 
For task 3, you can see all my codes in directory `Task3`. 

### Run
You can run the codes starting from root directory:
```bash
cd Task3
python GradCAM.py
python Shapley.py
python IntergratedGradients.py
```
### Files
```
- Task3
 |- GradCAM.py                  # main script of Grad-CAM.
 |- Shapley.py                  # main script of Shapley value.
 |- IntergratedGradients.py     # main script of Intergrated Gradients.
 |- dataset                     # a figure used in original paper of Grad CAM as a testset.
 |- output                      # (After you run all the three script, this directory will be automatically built, which contains visualization results)
    |- Task3_4                  # result of Grad-CAM.
    |- Task3_5                  # result of Shapley Value.
    |- Task3_6                  # result of Intergrated Gradients.
```