# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 09:36:39 2018

@author: htshi
""" 
import numpy as np
import pandas as pd
import math

##用于加载数据或生成数据等
import operator
class DecisionTree():
    #--------------------计算信息熵-----------------#
    def calculate_entropy(self,dataset):
        #获得样本容量|D| 
        datanum = len(dataset)
        #获得样本标签(类别)
        label=dataset.label.value_counts()
        label_arr=np.array(label.index)
        #初始化信息熵
        shannonEnt=0.0
        for i in label_arr:
            prob = label[i]/datanum
            shannonEnt -= prob * math.log(prob, 2)          
        return shannonEnt
    #--------------------划分数据集----------------#
    def split_dataset(self,dataset,feature_name,feature_index,value,head_arr):
        #获取标题头
        column=list(head_arr)
        #print("标题:",head_arr,"特征:",feature_name,"特征下标:",feature_index,"特征取值:",value)
        #若标题内有待划分特征，则删除
        if(column.count(feature_name)!=0):
            column.remove(feature_name) 
        #标题头添加'label'
        column.append('label')
        #选中这个特征列
        feature_col=dataset[feature_name]
        #初始化返回数据集列表
        ret_dataset=[]
        #根据取值划分数据集
        for i in range(len(dataset)):
            if(feature_col[i]==value):
                midvar=list(dataset.iloc[i])
                remaindata=midvar[:feature_index]
                remaindata.extend(midvar[feature_index+1:])
                ret_dataset.append(remaindata)
        #将列表转化为dataframe
        ret_dataset=pd.DataFrame(ret_dataset,columns=column)
        return ret_dataset #返回不含划分特征的子集
    
    
    def choose_best_feature_ID3(self,dataset,featureArr):
        feature_num=len(dataset.iloc[0])-1#特征数
        emp_entropy=self.calculate_entropy(dataset)#计算数据集D的经验熵
        
        bestInforGain=0#初始化最大信息增益
        bestFeature=''#初始化最佳划分特征
        bestFeatureNum=-1
        

        #-------------计算每个特征对数据集的经验条件熵--------------#
        for fea_index,fea_name in enumerate(featureArr):
            featList=dataset[fea_name] #某个特征的所有取值
            uniqualVals=set(featList) #set无重复的属性特征值
            
            emp_cond_entropy=0
            head_index=np.array(featureArr)
            
            names = locals()
            for value in uniqualVals:
                #对这个特征的所有可能取值进行划分数据集
                values_head=head_index
                subdataset=self.split_dataset(dataset,fea_name,fea_index,value,values_head)
                prob=len(subdataset)/float(len(dataset)) #即每个划分子集占整个子集的比例
                res=self.calculate_entropy(subdataset)
                emp_cond_entropy+=prob*res#self.calculate_entropy(subdataset)#对各子集经验熵求和得到经验条件熵
            infoGain=emp_entropy-emp_cond_entropy #计算信息增益
            #print("对于",fea_name,"特征：","条件经验熵为：",emp_cond_entropy,"信息增益为：",infoGain)
            #------得到最大信息增益和最佳划分特征-----#
            if (infoGain>bestInforGain):
                bestInforGain=infoGain
                bestFeature=fea_name
                bestFeatureNum=fea_index
        #print("最好的划分特征是：(choosebest)",bestFeature)

        return bestFeature,bestFeatureNum #返回最佳划分特征值
    def choose_best_feature_C45(self,dataset,featureArr):
        feature_num=len(dataset.iloc[0])-1#特征数
        emp_entropy=self.calculate_entropy(dataset)#计算数据集D的经验熵
        
        bestInforGain=0#初始化最大信息增益
        bestFeature=''#初始化最佳划分特征
        bestFeatureNum=-1
        

        #-------------计算每个特征对数据集的经验条件熵--------------#
        for fea_index,fea_name in enumerate(featureArr):
            featList=dataset[fea_name] #某个特征的所有取值
            uniqualVals=set(featList) #set无重复的属性特征值
            
            emp_cond_entropy=0
            splitInfo=0
            head_index=np.array(featureArr)
            
            names = locals()
            for value in uniqualVals:
                #对这个特征的所有可能取值进行划分数据集
                values_head=head_index
                subdataset=self.split_dataset(dataset,fea_name,fea_index,value,values_head)
                prob=len(subdataset)/float(len(dataset)) #即每个划分子集占整个子集的比例
                res=self.calculate_entropy(subdataset)
                emp_cond_entropy += prob*res#self.calculate_entropy(subdataset)#对各子集经验熵求和得到经验条件熵
                splitInfo -= prob * math.log(prob,2)
            infoGain=(emp_entropy-emp_cond_entropy)/splitInfo #计算信息增益率
            #print("对于",fea_name,"特征：","条件经验熵为：",emp_cond_entropy,"信息增益为：",infoGain)
            #------得到最大信息增益和最佳划分特征-----#
            if (infoGain>bestInforGain):
                bestInforGain=infoGain
                bestFeature=fea_name
                bestFeatureNum=fea_index
        #print("最好的划分特征是：(choosebest)",bestFeature)

        return bestFeature,bestFeatureNum #返回最佳划分特征值
    #--------------按分类后类别数量排序----------------------#
    def majorityCnt(self,classList):    
        classCount={}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote]=0
            classCount[vote]+=1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]
    #--------------构建决策树(传入数据集和特征值列表)----------------------#
    def create_tree_ID3(self,dataset,feat_labels):
            classList=list(dataset['label'])#获得类别列表

            #若是所有样本属于同一类别，则返回这一类别做节点标记，停止划分
            if classList.count(classList[0])==len(classList):
                #print("该节点上所有样本为同一类")
                return classList[0]

            #若特征集为空，则返回数据集中样本数最多的类作为节点标记，停止划分
            if len(dataset.iloc[0])==1:
                #print("特征集为空")
                return self.majorityCnt(classList)

            bestFeatLabel,bestFeatIndex=self.choose_best_feature_ID3(dataset,feat_labels) #选择最优特征

            print("最佳划分特征(createtree)",bestFeatLabel,bestFeatIndex)
            myTree={bestFeatLabel:{}} #分类结果以字典形式保存  
            #如果最佳划分特征不为空则继续划分
            if(bestFeatLabel!=''):

                del(feat_labels[bestFeatIndex])
                #print("最佳划分特征(createtree)",bestFeatLabel,bestFeatIndex)
                featValues=dataset[bestFeatLabel] #最好划分特征的所有取值
                uniqueVals=set(featValues)
                for value in uniqueVals:
                    subLabels=feat_labels[:]
                    values_head=subLabels
                    newdataset=self.split_dataset(dataset,bestFeatLabel,bestFeatIndex,value,values_head)
                    myTree[bestFeatLabel][value]=self.create_tree_ID3(newdataset,subLabels)
                return myTree

        
    #--------------C4.5构建决策树(传入数据集和特征值列表)----------------------#
    def create_tree_C45(self,dataset,feat_labels):
        #print("本节点特征",labels)
        classList=list(dataset['label'])#获得类别列表

        #若是所有样本属于同一类别，则返回这一类别做节点标记，停止划分
        if classList.count(classList[0])==len(classList):
            #print("该节点上所有样本为同一类")
            return classList[0]

        #若特征集为空，则返回数据集中样本数最多的类作为节点标记，停止划分
        if len(dataset.iloc[0])==1:
            #print("特征集为空")
            return self.majorityCnt(classList)

        bestFeatLabel,bestFeatIndex=self.choose_best_feature_C45(dataset,feat_labels) #选择最优特征

        print("最佳划分特征(createtree)",bestFeatLabel,bestFeatIndex)
        myTree={bestFeatLabel:{}} #分类结果以字典形式保存        
        #如果最佳划分特征不为空则继续划分
        if(bestFeatLabel!=''):
            del(feat_labels[bestFeatIndex])
            #print("最佳划分特征(createtree)",bestFeatLabel,bestFeatIndex)
            featValues=dataset[bestFeatLabel] #最好划分特征的所有取值
            #print(featValues)
            uniqueVals=set(featValues)
            for value in uniqueVals:
                subLabels=feat_labels[:]
                #print(bestFeatLabel,"取值为:",value)
                values_head=subLabels
                newdataset=self.split_dataset(dataset,bestFeatLabel,bestFeatIndex,value,values_head)
                #print(newdataset)
                #print(bestFeatLabel)
                myTree[bestFeatLabel][value]=self.create_tree_C45(newdataset,subLabels)
                #self.splitDataSet(dataset,bestFeatLabel,value),subLabels)
                #print(myTree)
            return myTree

    #使用决策树进行分类
    def classify(self, inputTree, featLabels, testVecs): 
        #获取根节点名
        firstStr = list(inputTree.keys())[0]  
        #获取除去根节点(该树首个划分特征)后的新树，仍表现为字典结构
        secondDict = inputTree[firstStr]
        #获取根节点(该树首个划分特征)在特征集中的索引
        featIndex = featLabels.index(firstStr)  
        for key in list(secondDict.keys()): 
            #根据索引对比测试数据相应位置的特征取值
            if testVecs[featIndex] == key:  
                if type(secondDict[key]).__name__ == 'dict':  #判断是否到了叶节点，若是到了叶节点则为str，若还有决策节点则为dict
                    classLabel = self.classify(secondDict[key], featLabels, testVecs)  #为决策节点则递归调用classify
                else: classLabel = secondDict[key]  
        return classLabel
    

        
