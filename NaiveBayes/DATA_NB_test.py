# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:19:52 2018

@author: htshinichi
"""
import NaiveBayesContinuous
import NaiveBayesDiscrete
import pandas as pd
from sklearn.model_selection import train_test_split

#------------------------------连续型-----------------------------------------#
diabetes = pd.read_csv("pima-indians-diabetes.csv")
dia_train,dia_test = train_test_split(diabetes,test_size=0.1)
model_NBC = NaiveBayesContinuous.NaiveBayesContinuous()
model_NBC.getPredictions(dia_train,dia_test)
acc1 = model_NBC.getAccuracy()
print(model_NBC.prediction)
print("%.2f" % acc1,"%")


#suika = pd.read_csv("suika4.0.csv")
#suika_train,suika_test = train_test_split(suika,test_size=0.2)
#model_NBC = NaiveBayesContinuous.NaiveBayesContinuous()
#model_NBC.getPredictions(suika_train,suika_test)
#acc2 = model_NBC.getAccuracy()
#print(model_NBC.prediction)
#print(acc2,"%")
#-----------------------------离散型------------------------------------------#
car = pd.read_csv("CarEvalution.csv")
car_train,car_test = train_test_split(car,test_size=0.1)
model_NBD = NaiveBayesDiscrete.NaiveBayesDiscrete()
acc2 = model_NBD.getAccuracy(car_train,car_test)
print()
print("%.2f" % acc2,"%")
