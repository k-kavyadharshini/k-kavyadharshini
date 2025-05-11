# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# STEP 2: Upload Dataset
print("Please upload 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")
uploaded = files.upload()

# STEP 3: Load Dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# STEP 4: Data Cleaning
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(inplace=True)

# STEP 5: Encode Categorical Columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Encode target column 'Churn'
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# STEP 6: Feature Scaling
X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# STEP 8: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# STEP 9: Evaluation
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 10: Visualization
df['Churn'].value_counts().plot(kind='bar', title='Churn Distribution', ylabel='Count', color=['skyblue', 'salmon'])
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.grid(axis='y')
plt.show()