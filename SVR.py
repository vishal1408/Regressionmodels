# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:31:11 2020

@author: chint
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2:3].values

from sklearn.preprocessing import StandardScaler
sx=StandardScaler()
sy=StandardScaler()
x=sx.fit_transform(x)
y=sy.fit_transform(y)

y=y.reshape(10)
from sklearn.svm import SVR
r=SVR(kernel='rbf')
r.fit(x,y)

y_predict=sy.inverse_transform(r.predict(sx.transform(np.array([[6.5]]))))

plt.scatter(sx.inverse_transform(x),sy.inverse_transform(y),color='red')
plt.plot(sx.inverse_transform(x),sy.inverse_transform(r.predict(x)),color='blue')
plt.title('Salaries v/s positions')
plt.xlabel('Poistions')
plt.ylabel('Salaries')
plt.show()
