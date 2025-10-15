import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('loan_data.csv')

# Data preprocessing
print("Preprocessing data...")

# Convert categorical variables to numerical
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Handle missing values
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# Convert Credit_History to numeric and handle missing values
df['Credit_History'] = pd.to_numeric(df['Credit_History'], errors='coerce')
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Split features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Convert target variable to binary
y = y.map({'Y': 1, 'N': 0})

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
print("Scaling numerical features...")
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Train the model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and preprocessing objects
print("\nSaving model and preprocessing objects...")
joblib.dump(model, 'loan_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel and preprocessing objects saved successfully!") 