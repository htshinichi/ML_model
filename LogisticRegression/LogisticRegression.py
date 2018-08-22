# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:07:45 2018

@author: htshinichi
"""
import random
import numpy as np  
import matplotlib.pyplot as plt
class LogisticRegression():
    def __init__(self,n_iter = 100,eta = 0.0001,gd = 'bgd',plot = False):
        self.n_iter = n_iter
        self.eta = eta
        self.gd = gd
        self.plot = plot
        
    def sigmoid(self,fx):
        return 1/(1+np.exp(-1 * fx))
    
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
                loss = hx - y_label
                gradient = np.dot(loss,data) / datanum
                self.weights = self.weights - self.eta * gradient#即weights = weights - eta*(h(x)-y)*x/datanum
                if n % 100 ==0:
                    self.cost = self.costFunction(np.array(TrainData[TrainData.columns.tolist()[0:featnum]]),y_label)
                    print("迭代",n,"次","损失值为：",self.cost)
                #可以选择训练过程中绘制决策边界
                if self.plot == True:
                    if n % (self.n_iter/5) == 0:
                        self.plotDecisionBoundary(TrainData)
            if self.plot == True:
                self.plotDecisionBoundary(TrainData)
                
            return self.weights
        
        #随机梯度下降
        if self.gd == 'sgd':
            flag = False
            for n in range(self.n_iter):
                if flag:
                    break
                x = random.randint(0,datanum-1)
                datax = np.array(TrainData[TrainData.columns.tolist()[0:featnum]])[x]
                datax_T = datax.transpose()
                hxx = self.sigmoid(np.dot(self.weights,datax_T))
                lossx = hxx-y_label[x]
                gradientx = lossx * datax
                self.weights = self.weights - self.eta * gradientx#即weights = weights - eta*(h(x)-y)*x/datanum
                if n % 100 ==0:
                    self.cost = self.costFunction(np.array(TrainData[TrainData.columns.tolist()[0:featnum]]),y_label)
                    print("迭代",n,"次","损失值为：",self.cost)
                #loss = 
                #可以选择训练过程中绘制决策边界
                if self.plot == True:
                    if n % (self.n_iter/5) == 0:
                        self.plotDecisionBoundary(TrainData)
            if self.plot == True:
                self.plotDecisionBoundary(TrainData)
                
            return self.weights
    
    def costFunction(self,data,label):
        datanum = len(data)
        h = self.sigmoid(np.dot(self.weights,data.transpose()))
        #计算损失函数
        cost = (-1/datanum)*(np.dot(np.log(h),label)+(np.dot(np.log(1-h),1-label)))
        return cost
          
    
    def predict(self,testData):
        flag = np.dot(self.weights,testData)
        if flag > 0:
            pred = 1
        else:
            pred = 0
        return pred
    
    def accuracy(self,TestData):
        num = 0
        for i in range(len(TestData)):
            temp = np.array(TestData.iloc[i][0:len(TestData.columns)-1]).reshape(len(TestData.columns)-1,1)
            if self.predict(temp)==TestData.label.tolist()[i]:
                num = num + 1
        return num/float(len(TestData))*100
    
    def plotDecisionBoundary(self,TrainData):
        plt.figure(figsize=(10,8))
        plt.xlim(-4,4)  #  设置x轴刻度范围
        plt.ylim(-4,4)  #  设置y轴刻度范围
        plt.xlabel('x1')   
        plt.ylabel('x2')
        plt.title('decision boundary') 
        x1 = np.arange(-4,4,1)
        x2 =-1 * self.weights[0][0] * x1 / self.weights[0][1]
        plt.scatter(TrainData[TrainData.columns[0]], TrainData[TrainData.columns[1]], c=TrainData['label'], s=30)
        plt.plot(x1,x2)
        plt.show()
