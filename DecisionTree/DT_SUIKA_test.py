# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:43:13 2018

@author: htshinichi
"""
import pandas as pd
import DecisionTREE
import DrawDecisionTREE
sika = pd.read_csv("sika3.0.csv")
sika_feature=sika.columns.values.tolist()[:8]
fealabel = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
DT_model = DecisionTREE.DecisionTree(split='C45')
DDT = DrawDecisionTREE.DrawDecisionTree()
sika_model = DT_model.create_tree(sika,sika_feature)
print(sika_model)
DDT.createPlot(sika_model)
