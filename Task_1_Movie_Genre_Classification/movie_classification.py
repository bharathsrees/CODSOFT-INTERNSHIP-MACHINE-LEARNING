import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the CSV files
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
solution_df = pd.read_csv('test_data_solution.csv')

# Extract features and labels
X_train = train_df['DESCRIPTION']
y_train = train_df['GENRE']
X_test = test_df['DESCRIPTION']
y_test = solution_df['GENRE']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(classification_report(y_test, y_pred, zero_division=0))

# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model and vectorizer
joblib.dump(model, 'movie_genre_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Function to predict genres for user input
def predict_genre(plot_summaries):
    vectorized_summaries = vectorizer.transform(plot_summaries)
    predictions = model.predict(vectorized_summaries)
    return predictions

# Get user input for plot summaries
def get_user_plots():
    print("Enter your movie plot summaries (type 'done' to finish):")
    plots = []
    while True:
        plot = input("Plot Summary: ")
        if plot.lower() == 'done':
            break
        plots.append(plot)
    return plots

# Get user plots and make predictions
user_plots = get_user_plots()
if user_plots:
    predicted_genres = predict_genre(user_plots)
    
    # Display the results
    for plot, genre in zip(user_plots, predicted_genres):
        print(f"Plot: {plot}\nPredicted Genre: {genre}\n")

    # Sample plot for predicted genres
    def plot_predicted_genres(predicted_genres):
       # Calculate the predicted genre distribution
        genre_counts = pd.Series(predicted_genres).value_counts()

# Plot the distribution of predicted genres
        plt.figure(figsize=(10, 6))
        sns.barplot(x=genre_counts.index, y=genre_counts.values, hue=genre_counts.index, palette='viridis', legend=False)
        plt.title("Distribution of Predicted Genres")
        plt.xlabel("Genre")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()

    # Plot the results
    plot_predicted_genres(predicted_genres)
else:
    print("No plots were provided.")
