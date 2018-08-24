# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 13:29:22 2018

@author: htshinichi
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class TwoDimensionPlot():
    #初始化
    def __init__(self,data,warr):
        self.data = data
        self.warr = warr
        self.calCost(self.data,self.warr)

        
    def calCost(self,data,warr):
        datanum = len(data)
        featnum = len(data.columns)-1
        label = np.array(data.label)#获取真实标签集合
        data = np.array(data[data.columns.tolist()[0:featnum]])
        self.w1 = []
        self.w2 = []
        for i in range(len(warr)):
            self.w1.append(warr[i][0])
            self.w2.append(warr[i][1])
        #为了防止迭代次数过多导致权重矩阵过多影响运算速度，只取前1000个权重
        if(len(warr) > 1000):
            self.w1 = self.w1[0:1000]
            self.w2 = self.w2[0:1000]
        self.w1,self.w2 = np.meshgrid(np.array(self.w1),np.array(self.w2))
        self.cost = 0
        for i in range(datanum):
            fx = self.w1 * data[i][0]+ self.w2 * data[i][1]
            hx = 1/(1+np.exp(-1 * fx))
            self.cost = self.cost + (label[i]*np.log(hx)+(1-label[i])*np.log(1-hx))/datanum
            if i % 100 == 0:
                print("迭代",i,"次","一共",datanum,"次","  请等待")
        self.regular_cost = np.abs(self.w1)+np.abs(self.w2)
    
    def plotContour(self):
        #self.calCost(data,warr)
        plt.figure(figsize=(8,8))
        plt.xlim(-1,2)
        plt.ylim(-2,1)
        P=plt.contour(self.w1,self.w2,self.cost,10)
        plt.clabel(P, alpha=0.75, cmap='jet',inline=1, fontsize=10)
        C=plt.contour(self.w1,self.w2,self.regular_cost,1)
        plt.clabel(C, alpha=0.75, cmap='jet',inline=1, fontsize=10)
        plt.show()
    
    def plot3D(self):
        #self.calCost(data,warr)
        fig = plt.figure()
        ax =fig.add_subplot(111,projection='3d')
        ax.plot_surface(self.w1,self.w2,self.cost,rstride=3,cstride=3,cmap=cm.jet)
        plt.show()
