# Logistic Regression – Breast Cancer Classification

## Project Description

This repository contains a machine learning project that applies **Logistic Regression** to classify tumors as **malignant** or **benign** using the **Breast Cancer Wisconsin Dataset** from `sklearn.datasets`.

## What I Did

### 1. Loaded the Dataset
- Used `load_breast_cancer()` from `sklearn.datasets`.
- Combined the features and target labels into a single DataFrame.

### 2. Data Preprocessing
- Separated features (`X`) and labels (`y`).
- Performed a train/test split using `train_test_split`.
- Standardized features using `StandardScaler`.

### 3. Model Training
- Trained a Logistic Regression model using `LogisticRegression` from scikit-learn.

### 4. Model Evaluation
- Evaluated performance using:
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
  - ROC-AUC score

### 5. Sigmoid Function Visualization
- Applied the sigmoid function to the model's raw output (`decision_function`).
- Visualized the predicted probabilities to show how the sigmoid maps values to [0, 1].

## Files Included

- `Task-4.ipynb`: Google colab containing code, outputs, and plots.
- `README.md`: Project overview and documentation.

## Requirements

This project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

