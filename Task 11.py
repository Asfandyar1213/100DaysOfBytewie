
#1. Predicting Diabetes Onset Using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score


data = pd.read_csv('diabetes.csv')


data = data.fillna(data.mean())

# Encode categorical variables
le = LabelEncoder()
data['categorical_feature'] = le.fit_transform(data['categorical_feature'])
# Separate features and target variable
X = data.drop('diabetes_onset', axis=1)
y = data['diabetes_onset']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic regression model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)



#2. Classifying Iris Species Using Decision Trees


data = pd.read_csv('iris.csv')

# Separate features and target variable
X = data.drop('species', axis=1)
y = data['species']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", confusion_matrix)
print("Accuracy:", accuracy)




#3. Predicting Titanic Survival Using Logistic Regression


data = pd.read_csv('titanic.csv')

imputer = Imputer(strategy='median')
data['Age'] = imputer.fit_transform(data[['Age']])  # Replace with appropriate imputation based on analysis

# Encode categorical variables (one-hot encoding for embarked and gender)
ohe = OneHotEncoder(sparse=False)
data_encoded = pd.concat([data, pd.DataFrame(ohe.fit_transform(data[['Embarked', 'Sex']]))], axis=1)
data_encoded.drop(['Embarked', 'Sex'], axis=1, inplace=True)  # Remove original columns

# Separate features and target variable
X = data_encoded.drop('Survived', axis=1)
