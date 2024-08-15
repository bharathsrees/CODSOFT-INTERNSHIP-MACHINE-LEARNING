import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
fraud_train = pd.read_csv('fraudTrain.csv')
fraud_test = pd.read_csv('fraudTest.csv')

# Feature Engineering: Extract features from date-time
fraud_train['trans_date_trans_time'] = pd.to_datetime(fraud_train['trans_date_trans_time'])
fraud_test['trans_date_trans_time'] = pd.to_datetime(fraud_test['trans_date_trans_time'])

fraud_train['hour'] = fraud_train['trans_date_trans_time'].dt.hour
fraud_train['day_of_week'] = fraud_train['trans_date_trans_time'].dt.dayofweek

fraud_test['hour'] = fraud_test['trans_date_trans_time'].dt.hour
fraud_test['day_of_week'] = fraud_test['trans_date_trans_time'].dt.dayofweek

# Convert categorical data into numerical data
fraud_train['gender'] = fraud_train['gender'].map({'M': 0, 'F': 1})
fraud_test['gender'] = fraud_test['gender'].map({'M': 0, 'F': 1})

# One-hot encoding for the 'category' column
fraud_train = pd.get_dummies(fraud_train, columns=['category'])
fraud_test = pd.get_dummies(fraud_test, columns=['category'])

# Align the train and test dataframes
fraud_train, fraud_test = fraud_train.align(fraud_test, join='left', axis=1, fill_value=0)

# Drop unnecessary columns
cols_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num']
X_train = fraud_train.drop(cols_to_drop + ['is_fraud'], axis=1)
y_train = fraud_train['is_fraud']
X_test = fraud_test.drop(cols_to_drop + ['is_fraud'], axis=1)
y_test = fraud_test['is_fraud']

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Initialize models with class weights
class_weight = 'balanced'

log_reg = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight=class_weight)
tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight=class_weight)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42, class_weight=class_weight)

# Train models
log_reg.fit(X_train_res, y_train_res)
tree_clf.fit(X_train_res, y_train_res)
rf_clf.fit(X_train_res, y_train_res)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_tree_clf = tree_clf.predict(X_test)
y_pred_rf_clf = rf_clf.predict(X_test)

# Evaluate models
print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

print("\nDecision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree_clf))
print(classification_report(y_test, y_pred_tree_clf))

print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_clf))
print(classification_report(y_test, y_pred_rf_clf))

# Confusion matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf_clf)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.show()
