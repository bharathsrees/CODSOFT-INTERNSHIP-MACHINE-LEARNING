# Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn for subscription-based services using historical data. The goal is to identify customers who are likely to churn and develop strategies to retain them.

## Dataset

- The dataset used is `Churn_Modelling.csv`, which includes features related to customer behavior and demographics.
- You can download the dataset from [this link](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction).

## Approach

1. **Data Preprocessing**:
   - Dropped unnecessary columns such as `RowNumber`, `CustomerId`, and `Surname`.
   - Encoded categorical variables like `Geography` and `Gender` using `LabelEncoder`.
   - Split the data into training and testing sets.
   - Scaled the features using `StandardScaler`.

2. **Modeling**:
   - Trained multiple classification algorithms:
     - **Logistic Regression**
     - **Random Forest**
     - **Gradient Boosting**
   - Selected the best-performing model based on accuracy.

3. **Evaluation**:
   - Evaluated models using accuracy, precision, recall, and F1-score metrics.
   - Performed hyperparameter tuning using GridSearchCV to optimize the best model.

## Results

- **Logistic Regression**:
  - Accuracy: `81.55%`

- **Random Forest**:
  - Accuracy: `86.70%`
  - Best Hyperparameters: `{'max_depth': 30, 'min_samples_split': 10, 'n_estimators': 200}`
  
- **Gradient Boosting**:
  - Accuracy: `86.60%`

The **Random Forest** model was the best-performing model with an accuracy of `86.70%`.

## Files

- `customer_churn_prediction.py`: Main code file for training and evaluating the model.
- `Churn_Modelling.csv`: Dataset used for the analysis.
- `output.txt`: Contains the results and performance metrics of the model.
