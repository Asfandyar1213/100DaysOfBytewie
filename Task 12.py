
#Task 1: Predicting Employee Attrition Using Logistic Regression


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

# Load dataset
data = pd.read_csv('employee_attrition.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
categorical_cols = ['Department', 'Gender', ...]
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
onehot = OneHotEncoder()
data = onehot.fit_transform(data)

# Standardize numerical features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split data into features and target variable
X = data[:, :-1]
y = data[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)


#Task 2: Classifying Credit Card Fraud Using Decision Trees



# Load dataset
data = pd.read_csv('credit_card_fraud.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Standardize features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split data into features and target variable
X = data[:, :-1]
y = data[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print('ROC AUC:', roc_auc)
print('Confusion Matrix:\n', conf_matrix)

