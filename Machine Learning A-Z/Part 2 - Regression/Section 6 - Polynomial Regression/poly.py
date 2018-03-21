import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

'''from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)'''

#get linear regression model and polynomial regression model
#then compare them. 

from sklearn.linear_model import LinearRegression

lr_predict = LinearRegression()
lr_predict.fit(X,y)

#now import polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Now visualize linear model
plt.scatter(X, y, color = 'red')
plt.plot(X, lr_predict.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Reg)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Now visualize polynomial model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (polynomial Reg)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predict using linear and polynomial. First with linear
lr_predict.predict(6.5)
#Output will show salary#With polynomical,
lin_reg_2.predict(poly_reg.fit_transform(6.5))
