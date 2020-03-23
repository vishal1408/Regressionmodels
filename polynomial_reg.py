# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 06:24:57 2020

@author: chint
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.preprocessing import PolynomialFeatures
p=PolynomialFeatures(degree= 4)
x_poly=p.fit_transform(x)

from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,r.predict(p.fit_transform(x)),color='blue')
plt.title("Position v/s salaries")
plt.xaxis("positions")
plt.yaxis("salaries")
plt.show()

#making the curve more smooother
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color='red')
plt.plot(x_grid,r.predict(p.fit_transform(x_grid)),color='blue')
plt.title("Position v/s salaries")
plt.xlabel("positions")
plt.ylabel("salaries")
plt.show()

x_ans=np.array(6.5)
x_ans=x_ans.reshape(1,1)
x_ans=p.fit_transform(x_ans)

##finding out the salary of the person expected to be in the middle of two levels

r.predict(x_ans)
