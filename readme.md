# Image Feature Extraction and Segmentation Using Pretrained CNN Models

## Overview

This repository contains the implementation of various deep learning models for image **feature extraction** and **segmentation** tasks using **pretrained Convolutional Neural Networks (CNNs)** and **MaskFormer** for instance segmentation. The following models are included:
- **DenseNet-121**
- **ResNet-50**
- **EfficientNet** (B0, B4, B5, B6, B7)
- **MaskFormer with EfficientNet-B7 Backbone** for image segmentation

Performance metrics such as **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** are also calculated for evaluation.

## Datasets

The following datasets are used in this repository:

1. **Shanghai Tech Dataset**: Used for crowd counting and density estimation tasks. (https://paperswithcode.com/dataset/shanghaitech) 
2. **UCF QNRF Dataset**: A large dataset for crowd counting, offering a diverse set of images. (https://www.crcv.ucf.edu/data/ucf-qnrf/)
3. **Mall Dataset**: A smaller dataset for crowd counting and density estimation in a mall setting. (https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)

## Models Implemented

The following pretrained CNN models are used for feature extraction:

1. **DenseNet-121**
2. **ResNet-50**
3. **EfficientNet-B0**
4. **EfficientNet-B4**
5. **EfficientNet-B5**
6. **EfficientNet-B6**
7. **EfficientNet-B7**

Additionally, **MaskFormer with EfficientNet-B7** is integrated for pixel-wise instance segmentation tasks.

## MaskFormer with EfficientNet-B7

MaskFormer is a state-of-the-art model for **instance segmentation**, and in this repository, we integrate it with the **EfficientNet-B7** backbone to improve segmentation performance. The model is used to detect and segment objects in images at the pixel level, making it useful for tasks like object detection and image understanding.


## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/akanksha2305/crowd-counting.git
cd crowd-counting
``` 
### 2. Install Dependencies

```bash 
pip install -r requirements.txt
```
## 3. Model Selection

You can choose from any of the pretrained models (**DenseNet**, **ResNet**, **EfficientNet**) or **MaskFormer** by specifying the model in the `main.py` script.

#### Example for DenseNet-121:

```python
from models.allmodels import load_densenet121

model = load_densenet121()
```
## Example for MaskFormer with EfficientNet-B7

```python
from models.maskformer import load_maskformer_with_efficientnet_b7

model = load_maskformer_with_efficientnet_b7()
```
### 5. Training and Evaluation
To run training and evaluate the model, execute the main.py script:

```bash
Copy code
python main.py --model efficientnet_b7 --dataset path_to_your_dataset
```
The main.py script allows you to specify which model to use. It automatically loads the dataset, trains the model, and evaluates it using MAE and MSE metrics.

### 6. Calculating Metrics
Metrics like MAE and MSE are computed in metrics.py. You can compute them after training by running:

```bash
python metrics.py --predictions path_to_predictions --ground_truth path_to_ground_truth
7. MaskFormer Segmentation
For instance segmentation, MaskFormer with EfficientNet-B7 backbone is integrated. Use the following to train MaskFormer:

```bash
python main.py --model maskformer --dataset path_to_your_dataset
This will run MaskFormer, which utilizes EfficientNet-B7 as the backbone for improved performance in pixel-wise segmentation tasks.
```
Example Usage
Here’s an example of how to use the DenseNet-121 model:

```python
Copy code
from models.allmodels import load_densenet121

model = load_densenet121()
```
###  Proceed with training, evaluation, etc.
Model Training and Evaluation Example:
```bash
python main.py --model densenet121 --dataset dataset/train
```
### Example for MaskFormer with EfficientNet-B7:
```bash
python main.py --model maskformer --dataset dataset/train
```
### Optimizer and Hyperparameters: 
For the models, AdamW is used as the optimizer with the following hyperparameters:
1. Learning rate: 1e-4
2. Weight decay: 1e-2
3. These can be adjusted in the main.py script if required.

### Performance Metrics
The following performance metrics are implemented for model evaluation:
1. Mean Absolute Error (MAE)
2. Mean Squared Error (MSE)


