import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_startups.csv')

'''Dummy variable trap : model cannot distinguish between effects of one dummy variable and 
another variable. Must be avoided. At max. there must be one variable less per value. 
This mean if there are 2 values i.e. male female, only use 1 dummy variable.'''

'''Building a model'''
'''step 1: select variables to be included in model. Only select important ones '''

''' 5 methods of building a model
1. All in
2. Backword elinimation
3. Forward selection
4. Bidirection elimination
5. Score comparision

2,3 and 4 are called stepwise regression

When to apply which model :
1. Prior knowledge, You have to, Preparing for backward elinimation '''
#X and y be independent and dependent variables respectively
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])
ohe = OneHotEncoder(categorical_features = [3])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:] #Dummy variable trap is removed


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

#Now, predict using lr
predict = lr.predict(X_test)

#Building optimal model using BACKWARD ELINIMATION

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

#according to backward elinimation, take variables which impact on model

X_opt = X[:, [0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 

#Based on X-opt we removed p>0.05
X_opt = X[:, [0,1,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,2,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Conclusion : Optimal variable is only R&D spend


