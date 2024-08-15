# Movie Genre Classification

## Project Overview

This project focuses on building a machine learning model to predict the genre of a movie based on its plot summary. Accurate genre classification aids in better content recommendation and organization.

## Dataset

- The dataset used in this project can be downloaded from [this link](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).
- It contains movie plot summaries and their corresponding genres.

## Approach

1. **Text Preprocessing**:
   - Techniques used: tokenization, stopword removal, and lemmatization to clean the text data.

2. **Feature Extraction**:
   - Applied **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert textual data into numerical features.

3. **Modeling**:
   - Trained the model using **Naive Bayes** classifier:
     - **Multinomial Naive Bayes**

4. **Evaluation**:
   - Evaluated the model using accuracy, precision, recall, and F1-score metrics.

## Results

- The **Multinomial Naive Bayes** model provided the best performance with an accuracy of **[Actual Accuracy]%**.
- Detailed performance metrics and confusion matrix are available in `output.txt`.

## Files

- `movie_classification.py`: Main code file for training and evaluating the model.
- `output.txt`: Contains the results and performance metrics of the model.

