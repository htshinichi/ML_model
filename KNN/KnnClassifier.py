# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:14:25 2018

@author: htshinichi
"""
import numpy as np
from collections import Counter

#---------------------KNN模型定义----------------------#
class KNN():
    #--------------初始化参数k(最近邻个数)------------#
    def __init__(self,k=5):
        self.k = k
    #------------------计算欧氏距离-----------------#
    def euclidean_distance(self,one_sample,X_train):#计算一个测试样本与所有训练样本的欧氏距离
        #将测试样本变成只有1行
        one_sample = one_sample.reshape(1,-1)
        X_train = X_train.reshape(X_train.shape[0],-1)
        #将测试样本沿y轴复制，使其行数等于训练样本行数
        X_test = np.tile(one_sample,(X_train.shape[0],1))
        #计算测试样本与每一个训练样本的欧氏距离
        distance_arr = np.sqrt(np.power(X_test - X_train,2).sum(axis=1))
        return distance_arr
    #-------------获取最近k个近邻的标签------------#
    def get_k_neighbor_labels(self,distances,y_train,k):
        k_neighbor_labels = []
        #按照距离排序选择前k个获取标签
        for distance in np.sort(distances)[:k]:
            label = y_train[distances==distance]
            k_neighbor_labels.extend(label)
        #将返回k个近邻标签列表转为数组，并只有一行
        return np.array(k_neighbor_labels).reshape(-1, )
    #-------------投票得到某样本的类别------------#
    def vote(self,one_sample,X_train,y_train,k):
        #获取测试集所有样本与训练集欧氏距离的数组
        Distances = self.euclidean_distance(one_sample,X_train)
        y_train = y_train.reshape(y_train.shape[0],1)
        #获取最近k个近邻的标签数组
        self.knn_labels_arr = self.get_k_neighbor_labels(Distances,y_train,k)
        #初始化标签和数
        find_label,find_count = 0,0
        #使用Counter统计每个标签出现次数，label为标签，count为标签出现次数
        for label,count in Counter(self.knn_labels_arr).items():
            if count > find_count:
                find_count = count
                find_label = label
        return find_label
    #-------------------预测---------------------#
    def predict(self,X_test,X_train,y_train):
        y_pred = []
        #获取每个测试样本的预测标签
        for sample in X_test:
            label = self.vote(sample,X_train,y_train,self.k)
            y_pred.append(label)
        return np.array(y_pred)
    #-----------------求预测精确度------------------#
    def accuracy(self,y,y_pred):
        y=y.reshape(y.shape[0],-1)
        y_pred=y_pred.reshape(y_pred.shape[0],-1)
        return np.sum(y==y_pred)/len(y)
