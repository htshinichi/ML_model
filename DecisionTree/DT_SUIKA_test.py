# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:43:13 2018

@author: htshi
"""
import pandas as pd
import DecisionTREE
import DrawDecisionTREE
suika = pd.read_csv("suika3.0.csv")
suika_feature=suika.columns.values.tolist()[:8]
fealabel = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
DT_model = DecisionTREE.DecisionTree()
DDT = DrawDecisionTREE.DrawDecisionTree()
suika_model = DT_model.create_tree_C45(suika,suika_feature)
print(suika_model)
DDT.createPlot(suika_model)
