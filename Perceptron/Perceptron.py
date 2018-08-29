# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 19:50:48 2018

@author: htshinichi
"""
import numpy as np
import random
#-------------------感知机之口袋算法(非线性)--------------------#
class PocketAlgorithm():
    #---------------初始化参数lr和n_iter-------------#
    #-- lr为learning rate 学习率，n_iter为迭代次数--#
    def __init__(self,lr=0.01,n_iter=100):
        self.lr = lr
        self.n_iter = n_iter
    #----------------随机梯度下降SGD----------------#
    def train_sgd(self,TrainData):
        label = np.array(TrainData.label) #训练数据集标签
        X_train = np.array(TrainData[TrainData.columns.tolist()[0:len(TrainData.columns) - 1]]) #训练数据集        
        self.weights = np.zeros(len(TrainData.columns)-1) #初始化权重
        self.bias = 0 #初始化偏置
        for _ in range(self.n_iter):
            i = random.randint(0,len(X_train)) #每次迭代随机选取一个样本进行参数更新
            if (np.dot(X_train[i],self.weights) + self.bias) * label[i] <= 0:#如果这个样本分类错误
                self.weights[0:] += self.lr * label[i] * X_train[i]#
                self.bias += self.lr * label[i]
        return self
     #----------------批量梯度下降BGD----------------#
    def train_bgd(self,TrainData):
        label = np.array(TrainData.label) #训练数据集标签
        X_train = np.array(TrainData[TrainData.columns.tolist()[0:len(TrainData.columns) - 1]]) #训练数据集

        self.weights = np.zeros(len(TrainData.columns)-1) #初始化权重
        self.bias = 0 #初始化偏置
        for _ in range(self.n_iter):
            for i in range(len(TrainData)):#每次迭代对所有样本进行更新
                if (np.dot(X_train[i],self.weights) + self.bias) * label[i] <= 0:
                    self.weights[0:] += self.lr * label[i] * X_train[i]
                    self.bias += self.lr * label[i]
        return self  

    #-----------------预测标记类别------------------#
    def predict(self,X_sample):
        if np.dot(X_sample,self.weights)+self.bias >= 0:
            pre=1
        if np.dot(X_sample,self.weights)+self.bias < 0:
            pre=-1
        return pre
    #-----------------求预测精确度------------------#
    def accuracy(self,TestData):
        label = np.array(TestData.label) #训练数据集标签
        X_test = np.array(TestData[TestData.columns.tolist()[0:len(TestData.columns) - 1]]) #训练数据集
        num = 0
        for i in range(len(TestData)):
            pred = self.predict(X_test[i])
            if pred==label[i]:
                num += 1
        acc = num/len(X_test)
        return acc
