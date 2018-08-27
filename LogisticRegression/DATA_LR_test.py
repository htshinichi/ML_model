# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:30:42 2018

@author: htshinichi
"""
import LogisticRegression     
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing       
import matplotlib.pyplot as plt
import TwoDimensionPlot
##################################模型1########################################
#导入数据集
pima = pd.read_csv('pima-indians-diabetes.csv')
pd0 = pima[pima.columns.tolist()[0:len(pima.columns)-1]]
pd1 = pima['label']
#归一化
pd0=pd.DataFrame(preprocessing.scale(pd0))
pima = pd.concat([pd0,pd1],axis=1)
pima_train,pima_test = train_test_split(pima,test_size=0.3)
#定义模型：迭代1000次，学习率默认为0.0001，训练方式随机梯度下降
model_lr1 = LogisticRegression.LogisticRegression(n_iter=1000,gd='sgd')
#训练模型
model_lr1.fit(pima_train)
print("模型1训练集精确率：",model_lr1.accuracy(pima_train))
print("模型1测试集精确率：",model_lr1.accuracy(pima_test))
print("模型1权重为：",model_lr1.weights[0])

    

#导入数据集
data = pd.read_csv('test.csv')
#划分数据集
data_train,data_test = train_test_split(data,test_size=0.3)

#############################模型2#############################################
##定义模型：迭代10000次，学习率为0.01，训练方式默认批量梯度下降，使用L2正则化
model_lr2 = LogisticRegression.LogisticRegression(eta=0.01,n_iter=10000,regular='l2')
model_lr2.fit(data_train)
print("模型2训练集精确率：",model_lr2.accuracy(data_train))
print("模型2测试集精确率：",model_lr2.accuracy(data_test))
print("模型2权重为：",model_lr2.weights[0])
model_lr2.plotDecisionBoundary(data_test)

##画出损失函数等值线
#draw = TwoDimensionPlot.TwoDimensionPlot(data_train,model_lr2.warr)
#draw.plotContour()
####################################模型3#######################################
##定义模型：迭代1000次，学习率为0.0001，训练方式随机梯度下降，不使用正则化
model_lr3 = LogisticRegression.LogisticRegression(n_iter=1000,gd='sgd',regular='none')
model_lr3.fit(data_train)
print("模型3训练集精确率：",model_lr3.accuracy(data_train))
print("模型3测试集精确率：",model_lr3.accuracy(data_test))
print("模型3权重为：",model_lr3.weights[0])
model_lr3.plotDecisionBoundary(data_test)

#################################模型4#########################################
##定义模型：迭代1000次，学习率为0.01，训练方式默认批量梯度下降，使用L1正则化，正则惩罚项为0.05，打印损失值和迭代次数
model_lr4 = LogisticRegression.LogisticRegression(n_iter=1000,eta=0.01,regular='l1',lamda=0.05,showcost=True)
model_lr4.fit(data_train)
print("模型4训练集精确率：",model_lr4.accuracy(data_train))
print("模型4测试集精确率：",model_lr4.accuracy(data_test))
print("模型4权重为：",model_lr4.weights[0])
model_lr4.plotDecisionBoundary(data_test)


