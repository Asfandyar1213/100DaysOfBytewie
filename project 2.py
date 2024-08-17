import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load the dataset
file_path = 'job skills.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
st.write("### Dataset Preview")
st.write(data.head())

# Display basic information about the dataset
st.write("### Dataset Information")
st.write(data.info())

# Display dataset statistics
st.write("### Dataset Statistics")
st.write(data.describe())

# Handle missing values if any
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the data into features and target variable
X = data.drop('target_column', axis=1)
y = data['target_column']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.write("### Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")
st.write("### Classification Report")
st.write(report)

# Define a function to predict job matching
def predict_job_match(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_df = scaler.transform(input_df)
    prediction = model.predict(input_df)
    return label_encoders['target_column'].inverse_transform(prediction)[0]

# Create the Streamlit UI
st.title("Job Skill Matching System")

# Input fields for job seeker information
st.sidebar.header("Input Job Seeker Information")
input_data = {}
for column in X.columns:
    if data[column].dtype == 'float64' or data[column].dtype == 'int64':
        input_data[column] = st.sidebar.slider(f"{column}", float(X[column].min()), float(X[column].max()), float(X[column].mean()))
    else:
        input_data[column] = st.sidebar.selectbox(f"{column}", options=data[column].unique())

# Button to predict job match
if st.sidebar.button("Predict Job Match"):
    result = predict_job_match(input_data)
    st.write(f"### Recommended Job: {result}")


