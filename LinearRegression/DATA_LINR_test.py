# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:45:43 2018

@author: htshinichi
"""
from matplotlib import pyplot as plt
import LinearRegression
import pandas as pd
import numpy as np

data = pd.read_csv("test_Regression.csv")
X=data.x1
y=data.label
model_linr = LinearRegression.LinearRegression()
model_linr.fit(data)
print(model_linr.weights)
print(model_linr.bias)
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = model_linr.predict(line_X)

plt.plot(line_X, line_y, color='navy', linewidth=2, label='Linear regressor')
plt.scatter(X,y,color='yellowgreen', marker='.',label='Inliers')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
