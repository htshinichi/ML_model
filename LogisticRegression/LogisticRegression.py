# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:07:45 2018

@author: htshinichi
"""
import random
import numpy as np  
import matplotlib.pyplot as plt
class LogisticRegression():    
    #初始化
    def __init__(self,n_iter = 100,eta = 0.0001,gd = 'bgd',plot = False,regular = 'none',lamda = 1,showcost = False):
        self.n_iter = n_iter #迭代次数
        self.eta = eta #学习率
        self.gd = gd #学习方式
        self.plot = plot #绘制决策边界变化
        self.regular = regular #正则化
        self.lamda = lamda #正则化惩罚系数
        self.showcost = showcost #显示迭代次数和cost值
        
    #对数几率函数    
    def sigmoid(self,fx):
        return 1/(1+np.exp(-1 * fx))
    
    #训练
    def fit(self,TrainData):
    
        self.datanum = len(TrainData)#获取训练集大小
        self.featnum = len(TrainData.columns)-1#获取特征数量        
        self.label = np.array(TrainData.label)#获取真实标签集合
        self.data = np.array(TrainData[TrainData.columns.tolist()[0:self.featnum]])#datanum×featnum
        self.data_T = self.data.transpose()#数据大小为featnum×datanum
        self.weights = np.zeros((1,self.featnum))#初始化权重，权重大小为1×featnum                
        self.warr = []
        self.carr = []
        #批量梯度下降
        if self.gd == 'bgd':
            for n in range(self.n_iter):
                #sigmoid([1×featnum] × [featnum×datanum]) --> [1×datanum]
                #预测值，即h(x)=1/(1+e^(-wx))
                hx = self.sigmoid(np.dot(self.weights,self.data_T)) 
                
                #[1×datanum] - [1×datanum] --> [1×datanum]
                #h(x)-y,误差值
                loss = hx - self.y_label 
                
                #L2正则化项求导,更新每个权重时都要加上，shape为[1×featnum]
                penalty = np.ones((1,self.featnum))*2*self.lamda*self.weights.sum()
                
                if self.regular == 'l2':
                    #([1×datanum] × [datanum×featnum] + [1×featnum])/datanum --> [1×featnum]
                    #([h(x)-y]x + 2*lamda*w)/m
                    gradient = (np.dot(loss,self.data) + penalty) / self.datanum 
                
                elif self.regular == 'none': 
                    #([1×datanum] × [datanum×featnum])/datanum --> [1×featnum]
                    #[h(x)-y]x/m
                    gradient = (np.dot(loss,self.data)) / self.datanum
                
                #[1×featnum] - eta*[1×featnum] --> [1×featnum]
                self.weights = self.weights - self.eta * gradient
                
                #显示迭代次数和损失值
                if (n % 100 == 0) & self.showcost:
                    self.cost = self.costFunction()
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
            for n in range(self.n_iter):
                #生成一个随机数x
                x = random.randint(0,self.datanum-1)  
                datax = self.data[x]  #[1×featnum]
                datax_T = datax.transpose()  #[featnum×1]
                
                #sigmoid([1×featnum] × [featnum×1]) --> [1×1]
                hxx = self.sigmoid(np.dot(self.weights,datax_T))
                
                #[1×1] - [1×1] --> [1×1]
                lossx = hxx-self.label[x]
                
                #L2正则化项求导,更新每个权重时都要加上，shape为[1×featnum]
                penalty = np.ones((1,self.featnum))*2*self.lamda*self.weights.sum() 
                
                    
                if self.regular == 'l2':
                    #([1×1] × [1×featnum] + [1×featnum]) --> [1×featnum]
                    gradientx = (lossx * datax) + penalty
                if self.regular == 'none':
                    #([1×1] × [1×featnum]) --> [1×featnum]
                    gradientx = lossx * datax
                    
                    
                #[1×featnum] - eta*[1×featnum] --> [1×featnum]
                self.weights = self.weights - self.eta * gradientx
                self.warr.append(self.weights[0].tolist())
                #当损失值小于0.1时停止迭代
                if self.costFunction() < 0.1:
                    break
                #显示迭代次数和损失值
                if (n % 100 == 0) & self.showcost:
                    self.cost = self.costFunction()
                    print("迭代",n,"次","损失值为：",self.cost)
                    
                #可以选择训练过程中绘制决策边界
                if self.plot == True:
                    if n % (self.n_iter/5) == 0:
                        self.plotDecisionBoundary(TrainData)
            if self.plot == True:
                self.plotDecisionBoundary(TrainData)
                
            return self.weights
    
    #计算损失值
    def costFunction(self):
        #sigmoid([1×featnum] × [featnum×datanum]) --> [1×datanum]
        h = self.sigmoid(np.dot(self.weights,self.data_T))
        
        #计算损失函数:E(w)=L(w)+lamda*R(w)
        #E(w) = -1/datanum * [y*ln(h(x)) + (1-y)*ln(1-h(x))]

        #R(w) = ||w||1 权重绝对值之和
        if self.regular == 'l1':
            #lamda*1 --> [1×1]
            C = self.lamda * np.abs(self.weights).sum()#L1正则化项
            
            #([1×datanum]×[datanum×1]) + ([1×datanum]×[datanum×1]) --> [1×1]
            cost = (-1/self.datanum) * (np.dot(np.log(h),self.label)+(np.dot(np.log(1-h),1-self.label)) + C)
            
        #R(w) = ||w||2 权重平方之和   
        elif self.regular == 'l2':
            #lamda * ([1×featnum] × [featnum×1]) --> [1×1]
            C = self.lamda * np.dot(self.weights,self.weights.transpose())#L2正则化项
            #([1×datanum]×[datanum×1]) + ([1×datanum]×[datanum×1]) --> [1×1]
            cost = (-1/self.datanum) * (np.dot(np.log(h),self.label)+(np.dot(np.log(1-h),1-self.label)) + C)            

        
        elif self.regular == 'none':
            cost = (-1/self.datanum) * (np.dot(np.log(h),self.label)+(np.dot(np.log(1-h),1-self.label)))
          
        return cost

    #预测单一样本
    def predict(self,testData):
        flag = np.dot(self.weights,testData)
        if flag > 0:
            pred = 1
        else:
            pred = 0
        return pred
    
    #获取数据集准确率
    def accuracy(self,TestData):
        num = 0
        for i in range(len(TestData)):
            temp = np.array(TestData.iloc[i][0:len(TestData.columns)-1]).reshape(len(TestData.columns)-1,1)
            if self.predict(temp)==TestData.label.tolist()[i]:
                num = num + 1
            acc = num / float(len(TestData)) * 100
        return str(acc)+"%"
    
    #绘制决策边界
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
    
