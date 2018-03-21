import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#X is matrix of independent variables
#Y is matrix of Dependent variables
#We predict Y on basis of X
dataset = pd.read_csv('Salary_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#Random_state = 0 indicates no randomized factors in algorithm

#Fitting Simple linear regression to Training set
from sklearn.linear_model import LinearRegression
#This class will help us to build models
regresor = LinearRegression() #Here we created a machine
regresor.fit(X_train, y_train) #Machine has learnt

#Now, We'll predict future values based on test set
#y_pred is predicted salary vector
y_pred = regresor.predict(X_test)

#Visualization of Trainning set data
plt.scatter(X_train, y_train, color='red') #Make scatter plot of real value
plt.plot(X_train, regresor.predict(X_train), color='blue') #Plot Predictions
plt.title('Salary vs Exprience(Training Set)')
plt.xlabel('Exprience(Years)')
plt.ylabel('Salary')
plt.show


#Visualization of Test set data
plt.scatter(X_test, y_test, color='red') #Make scatter plot of real value
plt.plot(X_train, regresor.predict(X_train), color='blue') #Plot Predictions
plt.title('Salary vs Exprience(Training Set)')
plt.xlabel('Exprience(Years)')
plt.ylabel('Salary')
plt.show






