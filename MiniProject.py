#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Data Collection - Simulated data for illustration
data = pd.read_csv('happiness_data.csv')
# Data Preprocessing
# Handle missing values and outliers, feature selection, and engineering
data = data.dropna()
X = data.drop(columns=['Student_ID', 'Happiness_Score'])
y = data['Happiness_Score']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model Development
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Testing the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

