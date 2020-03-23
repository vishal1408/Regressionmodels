# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:32:37 2020

@author: chint
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:44:10 2020

@author: chint
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('50_Startups.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,4].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
o=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=o.fit_transform(x)

x=x[:,1:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_predict=regressor.predict(x_test)

import statsmodels.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=np.array(x[:,[0,1,2,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=np.array(x[:,[0,1,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=np.array(x[:,[0,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=np.array(x[:,[0,3,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=np.array(x[:,[0,3]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()


#1st feature of x ---r and spend----is the only feature that makes most impat on the output!!!
x_new=x[:,3:4]


from sklearn.model_selection import train_test_split
xn_train,xn_test,yn_train,yn_test=train_test_split(x_new,y,test_size=0.2,random_state=0)
regressor.fit(xn_train,yn_train)

yn_predict=regressor.predict(xn_test)


