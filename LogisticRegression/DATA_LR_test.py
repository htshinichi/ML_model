# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:30:42 2018

@author: htshinichi
"""
import LogisticRegression     
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np       
import matplotlib.pyplot as plt

#导入数据集
pima = pd.read_csv('E:\Desktop\DataSet\pima-indians-diabetes.csv')
pd0 = pima[pima.columns.tolist()[0:len(pima.columns)-1]]
pd1 = pima['label']
#归一化
pd0=pd.DataFrame(preprocessing.scale(pd0))
pima = pd.concat([pd0,pd1],axis=1)
pima_train,pima_test = train_test_split(pima,test_size=0.3)
#定义模型：迭代1000次，学习率默认为0.0001，训练方式随机梯度下降
model_lr1 = LogisticRegression.LogisticRegression(n_iter=100000,gd='sgd')
#训练模型
model_lr1.fit(pima_train)
print("训练集精确率：",model_lr1.accuracy(pima_train))
print("测试集精确率：",model_lr1.accuracy(pima_test))
print("权重为：",model_lr1.weights[0])

#导入数据集
data = pd.read_csv('E://Desktop//DataSet//test.csv')
#划分数据集
data_train,data_test = train_test_split(data,test_size=0.3)
#定义模型：迭代次数默认100，学习率默认0.0001，训练方式默认批量梯度下降，显示决策边界变化情况
model_lr2 = LogisticRegression.LogisticRegression(eta=0.01,n_iter=10000,gd='bgd')
model_lr2.fit(data_train)
print("训练集精确率：",model_lr2.accuracy(data_train),"%")
print("测试集精确率：",model_lr2.accuracy(data_test),"%")
print("权重为：",model_lr2.weights[0])


fig=plt.figure(figsize=(10,8))
plt.xlim(-4,4)  #  设置x轴刻度范围
plt.ylim(-4,4)  #  设置y轴刻度范围
plt.xlabel('x1')   
plt.ylabel('x2')
plt.title('decision boundary') 
x1 = np.arange(-4,4,1)
x2 =-1 * model_lr2.weights[0][0] / model_lr2.weights[0][1] * x1
plt.scatter(data_train['x1'], data_train['x2'], c=data_train['label'], s=30, marker='o')
plt.scatter(data_test['x1'], data_test['x2'], c=data_test['label'], s=50, marker='*')
plt.plot(x1,x2)
plt.legend(["db","train","test"])
plt.show()
