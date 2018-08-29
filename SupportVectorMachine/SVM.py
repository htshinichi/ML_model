# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:35:32 2018

@author: htshinichi
"""

import numpy as np
import random
class SupportVectorMachine():
    def __init__(self,C=1,toler=0,kernelInfo=('linear',0),n_iter=100):
        self.C = C
        self.toler = toler
        self.kernelInfo = kernelInfo
        self.n_iter = n_iter
    
    #计算核函数
    def calKernelValues(self,data,datax):
        datalen = len(data)
        k_mat = np.zeros((datalen,1))#shape:(datalen,1)
        if self.kernelInfo[0] == 'linear':
            #线性核函数 K(x,z) = xz
            #data shape:(datalen,featnum)  datax.T shape :(featnum,)
            #(datalen,featnum) × (featnum,) --> (datalen,)
            k_mat = np.dot(data,datax.T)
            
        elif self.kernelInfo[0] == 'rbf':
            #径向基核函数 K(x,z) = exp((||x-z||^2)/(-2*delta^2))
            #循环后 --> [datalen×1]
            for j in range(datalen):
                
                deltaRow = data[j,:] - datax
                #[1×featnum] × [featnum×1] --> [1×1]
                #求出||x-z||^2 
                k_mat[j] = np.dot(deltaRow,deltaRow.T) 
                
            #[datalen×1] / [1×1] --> [datalen×1]
            #求出exp((||x-z||^2)/(-2*delta^2))
            k_mat = np.exp(k_mat/(-1*self.kernelInfo[1]**2)).reshape(datalen)
            
        else:
            raise NameError('That Kernel is not recognized!')
        
        return k_mat
    
    #对于给定的alphak计算差值Ek
    def calErrorValues(self,k):
        #计算alphaj * yj
        #[datanum×1] 对应相乘后转置 --> [1×datanum]
        alphay = np.multiply(self.alphas.T,self.label)
        
        #获取Kjk = K(xk,xj)，第k列
        #[datanum × 1] 
        Kjk = self.K[:,k]
        
        #g(xk) = sum(alphaj * yj) × K(xj,xk) + b
        #[1×datanum] × [datanum×1] + [1×1] --> [1×1]
        gxk = float(np.dot(alphay,Kjk) + self.b)
        
        #Ek = g(xk) - yk
        #[1×1] - [1×1] --> [1×1]
        Ek = gxk - float(self.label[k])
        
        return Ek         
    
    
    #存储差值
    def updateErrorCache(self,k):
        #计算第k个alpha的差值
        error = self.calErrorValues(k)
        #存入ErrorCacha，第一位的1表示这个差值是否有效
        self.ErrorCache[k] = [1,error]
    
    
    #随机选取下标不为i的j
    def selectRandj(self,i):
        j = i
        while(j==i):
            j = int(random.uniform(0,self.datanum))
        return j
        
    
    #调整大于H或小于L的alpha值
    def clipAlpha(self,aj,H,L):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj
    
    
    #选择第二个变量alphaj的差值Ej(传入第一个变量Ei和下标i)
    def selectAlphaj(self,i,Ei):
        #用maxk和maxEk表示使步长|Ei-Ej|最大的alphak的下标和Ek
        maxk,maxEk = -1,0
        #初始化Ej
        Ej = 0
        #将Ei在cache中设置为有效
        self.ErrorCache[i] = [1,Ei]
        #有效差值列表，用nonzero获取不为0的值的下标
        vaildErrorCacheList = np.nonzero(self.ErrorCache[:,0])[0]
        
        #若是有效差值列表中有有效差值(条件>1是因为该表中必然有一个有效差值,即Ei)
        if len(vaildErrorCacheList) > 1:
            #遍历有效差值列表
            for k in vaildErrorCacheList:
                #若找到Ei则跳过继续下一个
                if k == i: 
                    continue
                #计算Ek
                Ek = self.calErrorValues(k)                
                #计算△E=|Ei-Ek|
                deltaE = np.abs(Ei-Ek)
                if deltaE > maxEk:
                    maxk = k
                    maxEk = deltaE
                #找到最终使|Ei-Ek|最大的Ek赋值给Ej
                Ej = Ek
            return maxk,Ej
        
        #若是有效差值列表中没有有效差值(排除Ei)
        else:
            #在范围内随机选取一个j，计算Ej
            j = self.selectRandj(i)
            Ej = self.calErrorValues(j)
            
        return j,Ej 
    
    
    def violateKKT(self,Ei,i):
        if (self.label[i]*Ei<-1*self.toler)&(self.alphas[i]<self.C):
            return True
        elif (self.label[i]*Ei>self.toler)&(self.alphas[i]>0):
            return True
        else:
            return False


    def getb(self,Ei,Ej,i,j,alphai_old,alphaj_old):
        yiKii_alphai = self.label[i]*np.dot(self.TrainData[i,:],self.TrainData[i,:].T)*(self.alphas[i]-alphai_old)
        yjKji_alphaj = self.label[j]*np.dot(self.TrainData[j,:],self.TrainData[i,:].T)*(self.alphas[j]-alphaj_old)
        yiKij_alphai = self.label[i]*np.dot(self.TrainData[i,:],self.TrainData[j,:].T)*(self.alphas[i]-alphai_old)
        yjKjj_alphaj = self.label[j]*np.dot(self.TrainData[j,:],self.TrainData[j,:].T)*(self.alphas[j]-alphaj_old)
        b1 = self.b - Ei - yiKii_alphai - yjKji_alphaj
        b2 = self.b - Ej - yiKij_alphai - yjKjj_alphaj
        if (self.alphas[i]>0) & (self.alphas[i]<self.C):
            self.b = b1
        elif (self.alphas[j]>0) & (self.alphas[j]<self.C):
            self.b = b2
        else:
            self.b = (b1+b2)/2.0
    
    #内循环
    def InnerLoop(self,i):
        #计算第一个变量alphai的差值Ei
        Ei = self.calErrorValues(i)
        #选择违背KKT约束条件的alphai
        if self.violateKKT(Ei,i):
            j,Ej = self.selectAlphaj(i,Ei)
            alphai_old = self.alphas[i].copy()
            alphaj_old = self.alphas[j].copy()
            if self.label[i] != self.label[j]:
                #print(np.maximum(0,(self.alphas[j]-self.alphas[i])[0]))
                L = np.maximum(0,(self.alphas[j]-self.alphas[i])[0])
                H = np.minimum(self.C,self.C+self.alphas[j]-self.alphas[i])
            else:
                L = np.maximum(0,self.alphas[j]+self.alphas[i]-self.C)
                H = np.minimum(self.C,self.alphas[j]+self.alphas[i])
        
            if L==H:
                #print("L=H")
                return 0
            
            Kij = np.dot(self.TrainData[i,:],self.TrainData[j,:].T)
            Kii = np.dot(self.TrainData[i,:],self.TrainData[i,:].T)
            Kjj = np.dot(self.TrainData[j,:],self.TrainData[j,:].T)
            eta = 2.0 * Kij - Kii - Kjj            
            if eta >= 0:
                #print("eta>=0")
                return 0
            #print("eta:",eta)
            self.alphas[j] -= self.label[j] * (Ei-Ej) / eta
            #print("调整前aj:",self.alphas[j])
            self.alphas[j] = self.clipAlpha(self.alphas[j],H,L)
            #print("调整后aj:",self.alphas[j])
            self.updateErrorCache(j)
            
            if(np.abs(self.alphas[j]-alphaj_old)<0.000001):
                #print("j not moving enough")
                return 0
            
            self.alphas[i] = self.alphas[i] + self.label[j]*self.label[i]*(alphaj_old-self.alphas[j])
            self.updateErrorCache(i)
            
            self.getb(Ei,Ej,i,j,alphai_old,alphaj_old)

            return 1
        else:
            return 0

    
    def fit(self,TrainData):
        self.datanum = len(TrainData) #训练样本数量
        self.featnum = len(TrainData.columns) - 1 #特征数量
        self.TrainData =  np.array(TrainData[TrainData.columns.tolist()[0:self.featnum]]) #训练数据集
        self.label = np.array(TrainData.label) #训练数据集标签
        self.alphas = np.zeros((self.datanum,1))
        self.b = 0
        self.ErrorCache = np.zeros((self.datanum,2))
        self.K = np.zeros((self.datanum,self.datanum))
        for i in range(self.datanum):
            self.K[:,i] = self.calKernelValues(self.TrainData,self.TrainData[i,:]) #self.K[:,i] shape:(detanum)
        x_iter = 0
        entireSet = True
        alphaPairsChanged = 0 #遍历整个数据集修改任意alpha的次数
        #若
        while(x_iter<self.n_iter)&((alphaPairsChanged>0)|(entireSet))&(self.getAccuracy(TrainData)<0.9):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(self.datanum):
                    #若alphai被修改，则次数+1
                    alphaPairsChanged += self.InnerLoop(i)
                    #print("fullset,iter: %d i:%d,pairs changed %d" % (x_iter,i,alphaPairsChanged))
                x_iter += 1
            else:
                #获取0<alpha<C的坐标
                nonBoundIs = np.nonzero((self.alphas>0)&(self.alphas<self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.InnerLoop(i)
                    #print("non-bound,iter: %d i:%d,pairs changed %d" % (x_iter,i,alphaPairsChanged))
                x_iter += 1
            print("iter:",x_iter)
            print("修改次数",alphaPairsChanged)
            print("训练集准确度:",self.getAccuracy(TrainData))
            #print(entireSet)
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
                
        return self.b,self.alphas
                
        
    def getAccuracy(self,TestData):
        num = len(TestData)
        test_label = np.array(TestData.label)
        test_data =  np.array(TestData[TestData.columns.tolist()[0:self.featnum]])
        supportVectorsIndex = np.nonzero(self.alphas>0)[0]
        supportVectors = self.TrainData[supportVectorsIndex]
        supportVectorLabels = self.label[supportVectorsIndex]
        supportVectorAlphas = self.alphas[supportVectorsIndex]
        matchCount = 0
        for i in range(num):  
            kernelValue = self.calKernelValues(supportVectors, test_data[i,:])  
            #print("1",kernelValue)
            #print("2",np.multiply(supportVectorLabels, supportVectorAlphas.T)[0])
            predict = np.dot(kernelValue.T,np.multiply(supportVectorLabels, supportVectorAlphas.T)[0]) + self.b 
            if (np.sign(predict) == np.sign(test_label[i])):  
                matchCount += 1  
        accuracy = float(matchCount) / num  
        return accuracy  
