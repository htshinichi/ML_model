# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:15:01 2018

@author: htshinichi
"""

import numpy as np

class NaiveBayesContinuous():
    #获取训练集每个特征的均值和方差以及类标签的取值集合
    def getMeanStdLabel(self,train_data):
        label_counts=train_data.label.value_counts()
        label_arr=np.array(label_counts.index)
        label_arr.sort()
        #得到除标签外特征数
        num_feature = len(train_data.columns.values) - 1
        #按类别划分数据
        names = locals()
        for i in range(len(label_arr)):
            names['c%s' % i] = train_data[train_data["label"]==label_arr[i]]
        #按类别对每个属性求均值和方差
        c_mean=[]
        c_std=[]
        for j in range(len(label_arr)):
            names['mc%s' % j] = []
            names['sc%s' % j] = []
            for k in range(num_feature):
                names['mc%s' % j].append(np.mean(names['c%s' % j][k]))
                names['sc%s' % j].append(np.std(names['c%s' % j][k],ddof=1))
        
        for x in range(len(label_arr)):
            c_mean.append(names['mc%s' % x])
            c_std.append(names['sc%s' % x])
            names['arr_c%s' % x] = np.array(names['c%s' % x])
        return c_mean,c_std,label_arr
    #计算高斯概率密度函数
    def CalcuGaussProb(self,x,mean,stdev):
        exponent = np.exp(-(np.power(x-mean,2))/(2*np.power(stdev,2)))
        GaussProb = (1/(np.sqrt(2*np.pi)*stdev))*exponent
        return GaussProb
    
    #计算连续数据所属类的概率
    def CalcuClassProbCon(self,arr,cx_mean,cx_std):
        cx_probabilities=1
        for i in range(len(cx_mean)):
            cx_probabilities *= self.CalcuGaussProb(arr[i],cx_mean[i],cx_std[i])
        return cx_probabilities
    
    #单一样本预测
    def predict(self,trainData,testData):
        prob = []
        #print(trainData)
        self.cmean,self.cstd,self.label_array=self.getMeanStdLabel(trainData)
        for i in range(len(self.cmean)):
            cx_mean = self.cmean[i] #x类的均值
            cx_std = self.cstd[i] #x类的方差
            #print(testData)
            prob.append(self.CalcuClassProbCon(testData,cx_mean,cx_std)) #将计算得到的各类别概率存入列表
        bestLabel,bestProb = None,-1 #初始化最可能的类和最大概率值    
        for i in range(len(prob)): #找到所有类别中概率值最大的类
            if prob[i] > bestProb:
                bestProb = prob[i]
                bestLabel = self.label_array[i]
        return bestLabel,bestProb
    
    #整个数据集预测
    def getPredictions(self,TrainData,TestData):
        self.prediction = []
        self.testdata = np.array(TestData)
        for i in range(len(self.testdata)):
            result,proby = self.predict(TrainData,self.testdata[i])
            self.prediction.append(result)
        return self.prediction
    
    #计算准确性
    def getAccuracy(self):
        correct = 0
        for i in range(len(self.testdata)):
            if(self.testdata[i][-1]==self.prediction[i]):
                correct += 1
        return (correct/float(len(self.testdata)))*100.0
