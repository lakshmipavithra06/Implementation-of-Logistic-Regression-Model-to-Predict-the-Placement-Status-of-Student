# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and preprocess student data (marks, skills, etc.).
 
2. Split the dataset into training and testing sets.
 
3. Train the Logistic Regression model using training data.
  
4. Predict placement status on test data and evaluate accuracy.
    

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:LAKSHMI PAVITHRA M 
RegisterNumber: 212225220055

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


data = pd.read_csv("Placement_Data.csv")


data = data.drop("salary", axis=1)
data = pd.get_dummies(data, drop_first=True)

X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


print("Accuracy:", model.score(X_test, y_test))


y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()


print("\nClassification Report:\n", classification_report(y_test, y_pred))



X1 = X.iloc[:, 0].values.reshape(-1, 1)

model_plot = LogisticRegression(max_iter=1000)
model_plot.fit(X1, y)

plt.scatter(X1, y, color='blue')

x_values = np.linspace(X1.min(), X1.max(), 100)
y_values = model_plot.predict_proba(x_values.reshape(-1, 1))[:, 1]

plt.plot(x_values, y_values)

plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression Curve")
plt.show()

 
*/
```

## Output:

<img width="1363" height="754" alt="image" src="https://github.com/user-attachments/assets/812cbb04-be7b-4f3c-b160-0f7e1cbff550" />

<img width="1384" height="765" alt="image" src="https://github.com/user-attachments/assets/e4871932-7f55-420e-a80a-d7aed5523196" />

<img width="1245" height="658" alt="image" src="https://github.com/user-attachments/assets/8c3fb794-5136-4c6e-aa5d-4072b5e79b36" />

<img width="902" height="288" alt="image" src="https://github.com/user-attachments/assets/78e598b4-2f2e-44d5-b6e6-aafd46f510bd" />

<img width="1317" height="574" alt="image" src="https://github.com/user-attachments/assets/9a87e51f-e013-4078-9964-230ca062f4fc" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
