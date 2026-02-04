# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the student placement dataset and preprocess the data

2.Split the dataset into training and testing sets

3.Train the Logistic Regression model using training data

4.Predict the placement status and evaluate the model using confusion matrix

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:T.Goshanrajan 
RegisterNumber:  212225040098
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data.csv")

# Convert categorical values to numeric
data['gender'] = data['gender'].map({'M': 1, 'F': 0})
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Independent and dependent variables
X = data[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]
y = data['status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Predicted Placement Status:", y_pred)

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

## Output:
<img width="1272" height="142" alt="image" src="https://github.com/user-attachments/assets/d2e5d71f-5f20-4b45-9d10-c7b5c330d3af" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
