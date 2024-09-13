# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: mohamed asil s

RegisterNumber: 212223040112

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:

### Placement Data
![Screenshot 2024-09-12 212355](https://github.com/user-attachments/assets/ec12a589-9210-45bb-aff0-d07e561ab281)


### Checking the null() function
![Screenshot 2024-09-12 212916](https://github.com/user-attachments/assets/b9cd2ebc-b224-434b-b025-eb96a43722db)


### Print Data:
![Screenshot 2024-09-12 215253](https://github.com/user-attachments/assets/769e6c7f-dd0b-4d8f-9322-00b5ebeab004)


### Y_prediction array
![Screenshot 2024-09-12 215333](https://github.com/user-attachments/assets/7a0175c0-fca7-4004-b5aa-d00f03a70feb)


### Accuracy value
![Screenshot 2024-09-12 215704](https://github.com/user-attachments/assets/d06f9387-80fc-4558-9ef0-5e2f3625ca45)


### Confusion array
![Screenshot 2024-09-12 215845](https://github.com/user-attachments/assets/e5d00e7b-80d1-4a6a-81ec-091afb296962)


### Classification Report
![image](https://github.com/user-attachments/assets/594a2efe-ae05-4049-a7b6-50c1d7c6499d)

### Prediction of LR
![image](https://github.com/user-attachments/assets/a8a51bbb-82e2-45d7-a4a5-5269e0739b75)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
