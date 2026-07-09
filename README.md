#Machine Learning Project
DATASET: https://ap.wps.com/cms/docs/d/cbCaesISp46jEoM8?sa=601.1370


#Problem Statement : Predict whether a loan applicant is likely to not repay or repay the loan based on their financial and personal information.

#Importing Libraries
import pandas as pd 
import numpy as np 

#Collect Dataset
data = pd.read_csv(r"C:\Users\ADMIN\Downloads\credit_risk_dataset.csv.zip")

#Load Dataset
df=pd.DataFrame(data)
print(df)


#Explore Dataset
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df["person_age"])
print(df.sort_values("person_income"))
print(df.sort_values("loan_amnt"))
print(df["person_income"].max())
print(df["person_income"].min())
print(df["person_income"].mean())
print(df["person_income"].sum())


#Data Cleaning
print(df.isnull())

#Feature Selection
X=df[['person_age','person_income','loan_amnt','loan_percent_income']]
Y=df['loan_status']


#Train_Test_Split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
    )

#1 : Model Selection
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()


#Model Training
model1.fit(X_train,Y_train)


#Model Prediction
prediction = model1.predict([[34,59000,1000,0.1]])
print("Predicted Repayment Of Loan:",prediction)

#2 : Model Selection
from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()

#Model Training
model2.fit(X_train,Y_train)


#Model Prediction
prediction = model2.predict([[34,59000,1000,0.1]])
print("Predicted Repayment Of Loan:",prediction)

#3 : Model Selection
from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

#Model Training
model3.fit(X_train,Y_train)


#Model Prediction
prediction = model3.predict([[34,59000,1000,0.1]])
print("Predicted Repayment Of Loan:",prediction)

#Predictions On Test Data
# Logistic Regression
y_pred1 = model1.predict(X_test)

# Decision Tree
y_pred2 = model2.predict(X_test)

# Random Forest
y_pred3 = model3.predict(X_test)


#Import Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

#Evaluation Of Logistic Regression
print("Logistic Regression Performance:")

print("Accuracy :", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1))
print("Recall :", recall_score(Y_test, y_pred1))
print("F1 Score :", f1_score(Y_test, y_pred1))


#Evaluation Of Decision Tree Classifier
print("Decision Tree Classifier Performance:")

print("Accuracy :", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2))
print("Recall :", recall_score(Y_test, y_pred2))
print("F1 Score :", f1_score(Y_test, y_pred2))


#Evaluation Of Random Forest Classifier
print("Random Forest Classifier Performance:")

print("Accuracy :", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3))
print("Recall :", recall_score(Y_test, y_pred3))
print("F1 Score :", f1_score(Y_test, y_pred3))
