# Task 2: Credit Card Fraud Detection

## Project Overview

This project aims to detect fraudulent credit card transactions using machine learning. Fraud detection is a critical task in financial institutions to minimize loss and protect customers.

## Dataset

- The dataset consists of transaction records, including features such as transaction amount, time, and anonymized numerical features. It is available in `fraudTrain.csv` and `fraudTest.csv`.
- You can download the dataset from [this link](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

## Approach

1. **Data Preprocessing**:
   - Extracted additional features like the hour of the transaction and the day of the week.
   - Converted categorical features, such as gender and transaction category, into numerical form.
   - Handled class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).

2. **Feature Engineering**:
   - Performed feature scaling using `StandardScaler`.

3. **Modeling**:
   - Applied various classification algorithms with balanced class weights:
     - **Logistic Regression**
     - **Decision Trees**
     - **Random Forests**

4. **Evaluation**:
   - Evaluated models based on accuracy, precision, recall, F1-score, and the ROC-AUC curve.
   - Confusion matrices were also analyzed to understand model performance.

## Results

- **Logistic Regression**:
  - Accuracy: `99.37%`
  - Precision, Recall, F1-score: See detailed report in `output.txt`.

- **Decision Tree Classifier**:
  - Accuracy: `98.10%`
  - Precision, Recall, F1-score: See detailed report in `output.txt`.

- **Random Forest Classifier**:
  - Accuracy: `97.38%`
  - Precision, Recall, F1-score: See detailed report in `output.txt`.
  
  The **Logistic Regression** model achieved the best performance with an accuracy of `99.37%`.

## Files

- `fraud_detection.py`: Code for training and testing the models.
- `fraudTrain.csv` and `fraudTest.csv`: The datasets used for analysis.
- `output.txt`: Performance metrics of the different models.
