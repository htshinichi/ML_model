# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:16:51 2018

@author: htshinichi
"""

import KnnFunction
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

#加载digits数据集
digits = datasets.load_digits()
digits_X = digits.data   ##获得数据集输入
digits_y = digits.target ##获得数据集标签
#划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(digits_X, digits_y, test_size = 0.3)
k_value = np.arange(1,22,2).tolist()
##加载最近邻模型
KF = KnnFunction.KnnFunction()
KF.K_acc(X_train,X_test,y_train,y_test,k_value)
KF.plot_K_acc()
print("精确率最高的k值：",KF.k_best,"精确率：",KF.accu_best)
