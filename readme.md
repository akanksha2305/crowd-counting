# MFENet: A MaskFomer EfficientNet Instance Segmentation Approach for Crowd Counting

## Overview

This repository contains the implementation of MaskFormer architecture using EfficientNet-B7 as backbone for crowd counting. The following tasks are performed:
* MaskFormer with EfficientNet-B7 is integrated for pixel-wise instance segmentation tasks.
* Comparison of EfficientNet-B7 with EfficientNet-B0.B4.B5.B6.B7, DenseNet121 and ResNet50.
* Performance metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE) are used for evaluation.
  
## Datasets

The following datasets are used in this repository:

1. **ShanghaiTech Dataset**: Crowd Counting and density estimation tasks. Link for Dataset: https://paperswithcode.com/dataset/shanghaitech
2. **UCF-QNRF Dataset**: A large dataset for crowd counting, offering a diverse set of images. Link for Dataset: https://www.crcv.ucf.edu/data/ucf-qnrf/
3. **Mall Dataset**: A smaller dataset for crowd counting and density estimation in a mall setting. Link for Dataset: https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html

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

By specifying the model in the ' main.py ' script, you can choose from any of the pre-trained models (**DenseNet**, **ResNet**, **EfficientNet**) or **MaskFormer**.

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

### 6. MaskFormer Segmentation
For instance segmentation, MaskFormer with EfficientNet-B7 backbone is integrated. To train the MaskFormer: 

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


