# Attrition Prediction and Department Classification

## Overview

This repository contains Python code for building and evaluating a deep learning model to predict employee attrition and classify department based on various employee attributes.

## Dataset

The dataset used (`attrition.csv`) includes the following columns:
- Age
- Attrition
- BusinessTravel
- Department
- DistanceFromHome
- Education
- EducationField
- EnvironmentSatisfaction
- HourlyRate
- JobInvolvement
- ... (and more)

## Part 1: Preprocessing

### Dependencies
- `sklearn.model_selection.train_test_split`
- `sklearn.preprocessing.StandardScaler`
- `pandas`
- `numpy`
- `tensorflow.keras`

### Steps
1. **Importing Data**: Load the dataset from an external CSV file.
2. **Selecting Features**: Choose relevant columns (`X`) and target variables (`y`) for department and attrition prediction.
3. **Data Preprocessing**: 
   - Convert non-numeric data (`OverTime`) to numeric using lambda functions.
   - Scale numeric data using `StandardScaler`.
   - Encode categorical variables (`Department` and `Attrition`) using `OneHotEncoder`.

## Part 2: Model Creation and Training

### Model Architecture
- **Input Layer**: Accepts input features.
- **Shared Layers**: Two dense layers (`shared_layer_1` and `shared_layer_2`) common to both branches.
- **Department Branch**: Predicts department with a softmax output.
- **Attrition Branch**: Predicts attrition with a softmax output.

### Training
- **Compilation**: Uses categorical cross-entropy loss and Adam optimizer for both outputs.
- **Training**: Trains the model with 100 epochs and a batch size of 32.

### Evaluation
- **Evaluation Metrics**: Computes accuracy for both department and attrition predictions on the test set.

## Results
- **Accuracy**:
  - Department predictions: 51.7%
  - Attrition predictions: 80.6%

---

This `README.md` provides an overview of the dataset, preprocessing steps, model architecture, training process, evaluation metrics, and results obtained from running the notebook.

Feel free to customize it further based on additional insights or specific details about the dataset and methodology used!
