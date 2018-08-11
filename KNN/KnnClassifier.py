import numpy as np
import pandas as pd 
from collections import Counter
class KNN():
    #--------------初始化参数k(最近邻个数)------------#
    def __init__(self,k=5):
        self.k = k
    #------------------计算欧氏距离-----------------#
    def euclidean_distance(self,one_sample,X_train):#计算一个样本与训练中所有样本的欧氏距离的平方
        one_sample = one_sample.reshape(1,-1)
        X_train = X_train.reshape(X_train.shape[0],-1)
        distances = np.power(np.tile(one_sample,(X_train.shape[0],1)) - X_train,2).sum(axis=1)
        return distances
    #-------------获取最近k个近邻的标签------------#
    def get_k_neighbor_labels(self,distances,y_train,k):
        k_neighbor_labels = []
        for distance in np.sort(distances)[:k]:
            label = y_train[distances==distance]
            k_neighbor_labels.extend(label)
        return np.array(k_neighbor_labels).reshape(-1, )
    #-------------投票得到某样本的类别------------#
    def vote(self,one_sample,X_train,y_train,k):
        distances = self.euclidean_distance(one_sample,X_train)
        y_train = y_train.reshape(y_train.shape[0],1)
        k_neighbor_labels = self.get_k_neighbor_labels(distances,y_train,k)
        find_label,find_count = 0,0
        for label,count in Counter(k_neighbor_labels).items():
            if count > find_count:
                find_count = count
                find_label = label
        return find_label
    #-------------------预测---------------------#
    def predict(self,X_test,X_train,y_train):
        y_pred = []
        for sample in X_test:
            label = self.vote(sample,X_train,y_train,self.k)
            y_pred.append(label)
        return np.array(y_pred)
    #-----------------求预测精确度------------------#
    def accuracy(self,y,y_pred):
        y=y.reshape(y.shape[0],-1)
        y_pred=y_pred.reshape(y_pred.shape[0],-1)
        return np.sum(y==y_pred)/len(y)
