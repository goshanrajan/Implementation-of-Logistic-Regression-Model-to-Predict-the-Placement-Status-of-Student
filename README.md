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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

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

# ---------------- USER INPUT ----------------
print("\nEnter Student Details:")

gender = input("Gender (M/F): ")
gender = 1 if gender.upper() == 'M' else 0

ssc_p = float(input("SSC Percentage: "))
hsc_p = float(input("HSC Percentage: "))
degree_p = float(input("Degree Percentage: "))
etest_p = float(input("E-Test Percentage: "))
mba_p = float(input("MBA Percentage: "))

user_data = pd.DataFrame([[gender, ssc_p, hsc_p, degree_p, etest_p, mba_p]],
                         columns=['gender', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'])

user_prediction = model.predict(user_data)

if user_prediction[0] == 1:
    print("\nPlacement Status: PLACED")
else:
    print("\nPlacement Status: NOT PLACED")

# ---------------- MODEL EVALUATION ----------------
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Sensitivity (Recall)
sensitivity = recall_score(y_test, y_pred)
print("Sensitivity (Recall):", sensitivity)

# Specificity
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)
print("Specificity:", specificity)
```

## Output:
<img width="706" height="432" alt="image" src="https://github.com/user-attachments/assets/388f919c-3536-4b05-a9bf-a058f7268447" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
