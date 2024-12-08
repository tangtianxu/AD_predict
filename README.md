# Alzheimer Disease Prediction - MLP Model

This project aims to predict Alzheimer's disease using a Multi-Layer Perceptron (MLP) model. The pipeline consists of several stages including data preprocessing, model design, model training, and evaluation. The main files in this repository are:

- `preprocess.py`: Used for data cleaning and preprocessing.
- `MLP_model.py`: Contains the definition of the MLP model architecture.
- `train.py`: Responsible for training the model, validating, and testing.
- `model/`: Folder for saving and loading the trained model weights.
- `data/`: Folder for saving and loading the original and preprocessed dataset.
## How to use
- install the required dependencies
- run `python preprocess.py`
- run `python train.py`


## File Overview

### 1. `preprocess.py`

This script performs data cleaning and preprocessing, including:
- Loading the raw data.
- Handling missing values.
- Feature scaling and encoding.


### 2. `MLP_model.py`

This file contains the definition of the Multi-Layer Perceptron (MLP) architecture for the model. The model is designed to predict Alzheimer's disease based on the features in the dataset. It includes:
- Model architecture (input layer, hidden layers, and output layer).
- Activation functions and batch normalization layers.

### 3. `train.py`

This script is responsible for:
- Splitting the dataset into training, validation, and test sets.
- Training the model.
- Validating the model on the validation set.
- Evaluating the model on the test set.
- Saving the best-performing model weights to the `model/` folder.

### 4. `model/` Folder

This folder is where the model weights are saved after training. The model is saved as a `.pth` file and can be loaded for inference or further evaluation.

### 5. `data/` Folder

This folder is where the original  and preprocessed data are saved. The dataset is saved as a `.csv` or `.npy` file and can be loaded by `preprocess.py` and `train.py`

## Setup Instructions

### Requirements

Before running the code, make sure you have the following libraries installed:
- Python 3.x

- `sklearn` (for data processing and metrics)
- `pandas` `numpy` `torch` (for data manipulation)
- `matplotlib` (for plotting)
- `seaborn` (for visualization of confusion matrix)

  
You can install the required dependencies using `pip`:
```bash
pip install torch scikit-learn pandas matplotlib seaborn numpy
