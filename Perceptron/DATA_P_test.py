# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:27:03 2018

@author: htshinichi
"""
import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split

pima = pd.read_csv("pima-indians-diabetes.csv")
data_train,data_test = train_test_split(pima,test_size=0.1)
model_p = Perceptron.PocketAlgorithm()
model_p.train_bgd(data_train)
print(model_p.accuracy(data_test))
print("权重：",model_p.weights,"偏置：",model_p.bias)
