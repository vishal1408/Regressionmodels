# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:52:30 2020

@author: chint
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary v/s Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
