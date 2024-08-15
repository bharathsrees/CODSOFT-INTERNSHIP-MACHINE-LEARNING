# Movie Genre Classification

## Project Overview

This project focuses on building a machine learning model to predict the genre of a movie based on its plot summary. Accurate genre classification aids in better content recommendation and organization.

## Dataset

- The dataset contains movie plot summaries and their corresponding genres, provided in the files:
  - `train_data.csv`: Training data
  - `test_data.csv`: Test data
  - `test_data_solution.csv`: True genres for the test data

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

- The **Multinomial Naive Bayes** model achieved an accuracy of **52%**.
- Detailed performance metrics include:
  - **Precision**: Varies by genre
  - **Recall**: Varies by genre
  - **F1-Score**: Varies by genre
- A sample prediction from the model:
  - Plot: "A young orphaned girl discovers a magical world inside a wardrobe and helps to end the eternal winter caused by an evil queen."
    - Predicted Genre: **Drama**
  - Plot: "In a dystopian future, a group of rebels fights against a totalitarian regime that controls the world's resources and oppresses its citizens."
    - Predicted Genre: **Documentary**
- Detailed performance metrics and confusion matrix are available in `output.txt`.

## Files

- `movie_genre_classifier.py`: Main code file for training and evaluating the model.
- `train_data.csv`: Training dataset.
- `test_data.csv`: Test dataset.
- `test_data_solution.csv`: True labels for the test dataset.
- `output.txt`: Contains the results and performance metrics of the model.
