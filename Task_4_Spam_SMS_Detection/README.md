# Spam SMS Detection

## Project Overview

This project aims to classify SMS messages as spam or legitimate using natural language processing techniques. The goal is to build a model that can effectively filter out spam messages, helping to prevent unwanted communication.

## Dataset

- The dataset used is `spam_sms.csv`, which contains SMS messages and their corresponding labels (spam or ham). The dataset is sourced from [Kaggle's SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## Approach

1. **Text Preprocessing**:
   - Applied tokenization, stopword removal, and feature extraction using TF-IDF.

2. **Modeling**:
   - Experimented with multiple classification algorithms, including Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).

3. **Evaluation**:
   - Evaluated models using accuracy, precision, recall, and F1-score metrics to identify the best-performing model.

## Results

- The best-performing model was **SVM**, achieving an accuracy of **98%**.
- Detailed performance metrics and confusion matrix are available in `output.txt`.

## Files

- `spam_sms_detection.py`: Main code file for training and evaluating the model.
- `spam_sms.csv`: Dataset used for the analysis.
- `output.txt`: Contains the results and performance metrics of the model.
- `best_spam_classifier_model.pkl`: Saved model for spam detection (based on Logistic Regression).
- `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer used for text feature extraction.
