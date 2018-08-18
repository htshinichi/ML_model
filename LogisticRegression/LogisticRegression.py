# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:07:45 2018

@author: htshinichi
"""
import random
import numpy as np  
class LogisticRegression():
    def __init__(self,n_iter = 100,eta = 0.0001,gd = 'bgd'):
        self.n_iter = n_iter #迭代次数，默认为100
        self.eta = eta #学习率，默认为0.0001
        self.gd = gd #学习方式，默认为批量梯度下降
        
    def sigmoid(self,fx):
        return 1/(1+np.exp(-1 * fx)) #对数几率函数
    
    def fit(self,TrainData):
        y_label = np.array(TrainData.label)#获取真实标签集合
        datanum = len(TrainData)#获取数据集大小
        featnum = len(TrainData.columns)-1#获取特征数量
        self.weights = np.zeros((1,featnum))#初始化权重，权重大小为1×featnum
        
        #批量梯度下降
        if self.gd == 'bgd':
            data = np.array(TrainData[TrainData.columns.tolist()[0:featnum]])#datanum×featnum
            data_T = data.transpose()#数据大小为featnum×datanum
            for n in range(self.n_iter):
                hx = self.sigmoid(np.dot(self.weights,data_T))#预测值，即h(x)=1/(1+e^(-wx))
                self.weights = self.weights - self.eta * np.dot((hx - y_label),data)/datanum#即weights = weights - eta*(h(x)-y)*x/datanum
            return self.weights
        
        #随机梯度下降
        if self.gd == 'sgd':
            for n in range(self.n_iter):
                x = random.randint(0,datanum-1)
                datax = np.array(TrainData[TrainData.columns.tolist()[0:featnum]])[x]
                datax_T = datax.transpose()
                hxx = self.sigmoid(np.dot(self.weights,datax_T))
                self.weights = self.weights - self.eta * (hxx - y_label[x]) * datax#即weights = weights - eta*(h(x)-y)*x/datanum
            return self.weights
    
    #预测单个样本类别            
    def predict(self,testData):
        flag = np.dot(self.weights,testData)
        if flag > 0:
            pred = 1
        else:
            pred = 0
        return pred
    
    #计算准确率
    def accuracy(self,TestData):
        num = 0
        for i in range(len(TestData)):
            temp = np.array(TestData.iloc[i][0:len(TestData.columns)-1]).reshape(len(TestData.columns)-1,1)
            if self.predict(temp)==TestData.label.tolist()[i]:
                num = num + 1
        return (num/float(len(TestData)))*100.0
