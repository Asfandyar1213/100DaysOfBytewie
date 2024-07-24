import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


#1. Classifying Loan Status Using Decision Trees

data = pd.read_csv('loan_data.csv')

# Preprocessing

imputer = SimpleImputer(strategy='mean')
data.fillna(imputer.fit_transform(data), inplace=True)

# Encode categorical variables
le = LabelEncoder()
data['loan_grade'] = le.fit_transform(data['loan_grade'])

# Standardize numerical features
scaler = StandardScaler()
data[['loan_amount', 'interest_rate']] = scaler.fit_transform(data[['loan_amount', 'interest_rate']])

# Split data
X = data.drop('loan_status', axis=1)
y = data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))



#2. Predicting Hospital Readmission Using Logistic Regression


data = pd.read_csv('hospital_readmission.csv')


# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))



#3. Classifying Digit Images Using Decision Trees

# Load MNIST dataset
digits = load_digits()

# Preprocessing
X = digits.data
y = digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))




#4. Predicting Loan Approval Using Logistic Regression


data = pd.read_csv('loan_prediction.csv')


# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#5. Classifying Wine Quality Using Decision Trees

# Assuming you have a CSV file named 'wine_quality.csv'
data = pd.read_csv('wine_quality.csv')


# Model training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

