# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 20:31:14 2018

@author: htshinichi
"""
import KnnClassifier
##用于可视化图表
import matplotlib.pyplot as plt
class KnnFunction():
    #根据k取值列表，计算各取值的精确率
    def K_acc(self,X_train,X_test,y_train,y_test,k_arr):
        self.k_value = k_arr
        self.accu_value=[]
        self.accu_best,self.k_best = 0, 0
        for k in self.k_value:
            model = KnnClassifier.KNN(k)
            y_pred = model.predict(X_test,X_train,y_train)
            accu = model.accuracy(y_test,y_pred)
            if accu > self.accu_best:
                self.k_best = k
                self.accu_best = accu
            self.accu_value.append(accu)
    #绘制k-acc图
    def plot_K_acc(self):
        figure = plt.figure(figsize=(8,6))
        ax = figure.add_subplot(1,1,1)
        ax.plot(self.k_value,self.accu_value,color='red',marker='*',lw=1)
        plt.xticks(self.k_value, rotation=0) 
        ax.set_xlabel(r"k_value")
        ax.set_ylabel(r"acc")
        ax.set_title("k value and accuracy")
        plt.show()

