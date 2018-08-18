# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:30:42 2018

@author: htshinichi
"""
import LogisticRegression     
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
pima = pd.read_csv('pima-indians-diabetes.csv')
pd0 = pima[pima.columns.tolist()[0:len(pima.columns)-1]]
pd1 = pima['label']
#将训练数据做归一化处理
pd0=pd.DataFrame(preprocessing.scale(pd0))
pima = pd.concat([pd0,pd1],axis=1)
#划分数据集
pima_train,pima_test = train_test_split(pima,test_size=0.3)
#使用随机梯度下降训练
model_lr = LogisticRegression.LogisticRegression(gd='sgd')
model_lr.fit(pima_train)
#显示准确率
print(model_lr.accuracy(pima_test))
