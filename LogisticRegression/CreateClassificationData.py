# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:51:04 2018

@author: htshinichi
"""

import pandas as pd
from sklearn import datasets
##随机生成一组特征数量为2，样本数量为500的数据集
X, y = datasets.make_classification(n_samples=500,n_features=2, n_redundant=0, n_informative=1,n_clusters_per_class=1)
df1 = pd.DataFrame(X,columns=["x1","x2"])
df2 = pd.DataFrame(y,columns=["label"])
test = pd.concat([df1,df2],axis=1)
test.to_csv('E://Desktop//DataSet//test.csv',index=None)
