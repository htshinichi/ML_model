# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:41:11 2018

@author: htshi
"""
import matplotlib.pyplot as plt
from pylab import *  
class DrawDecisionTree():
    def InitPram(self):
        mpl.rcParams['font.sans-serif'] = ['SimHei'] #否则中文无法正常显示        
        self.decisionNode=dict(boxstyle='circle',fc='0.6',pad=0.3) #决策点样式
        self.leafNode=dict(boxstyle='square',fc='0.6',pad=0.3)#叶节点样式
        self.arrow_args=dict(arrowstyle='<-') #箭头样式
        
    def plotNode(self,nodeTxt,centerPt,parentPt,nodeType):
        self.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,arrowprops=self.arrow_args)
    #获取叶节点数量（广度）
    #例：
    #{'纹理': {'模糊': '坏瓜', 
    #         '清晰': {'触感': {'软粘': {'密度': {0.40299999999999997: '好瓜', 
    #                                            0.24299999999999999: '坏瓜', 
    #                                            0.35999999999999999: '坏瓜'}}, 
    #                          '硬滑': '好瓜'}}, 
    #         '稍糊': {'触感': {'软粘': '好瓜', '硬滑': '坏瓜'}}}}
    def getNumLeafs(self,myTree):
        #初始化叶节点总数
        numLeafs=0
        firstStr=list(myTree.keys())[0]
        secondDict=myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':#str为叶节点即类似'xxx'，dict为决策节点即类似{'aaa':'bbb'}
                numLeafs+=self.getNumLeafs(secondDict[key])#若为决策节点则递归调用getNumLeafs
            else:numLeafs+=1#否则叶节点数+1
        return numLeafs
    
    #获取树的深度的函数（深度）逻辑与上面类似
    def getTreeDepth(self,myTree):
        #初始化最大深度
        maxDepth=0
        firstStr=list(myTree.keys())[0]
        secondDict=myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                thisDepth=1+self.getTreeDepth(secondDict[key])
            else: thisDepth=1
            if thisDepth > maxDepth:#将当前深度与记录最大深度比较
                maxDepth=thisDepth
        return maxDepth
    
    #定义在父子节点之间填充文本信息的函数
    def plotMidText(self,cntrPt,parentPt,txtString):
        xMid=(parentPt[0]-cntrPt[0])/2+cntrPt[0]
        yMid=(parentPt[1]-cntrPt[1])/2+cntrPt[1]
        self.ax1.text(xMid,yMid,txtString)
    
    #定义树绘制的函数    
    def plotTree(self,myTree,parentPt,nodeTxt):
        numLeafs=self.getNumLeafs(myTree)
        depth=self.getTreeDepth(myTree)
        firstStr=list(myTree.keys())[0]
        cntrPt=(self.xOff+(1.0+float(numLeafs))/2/self.totalW,self.yOff)
        self.plotMidText(cntrPt,parentPt,nodeTxt)
        self.plotNode(firstStr,cntrPt,parentPt,self.decisionNode)
        secondDict=myTree[firstStr]
        self.yOff=self.yOff -1/self.totalD
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                self.plotTree(secondDict[key],cntrPt,str(key))
            else:
                self.xOff=self.xOff+1.0/self.totalW
                self.plotNode(secondDict[key],(self.xOff,self.yOff),cntrPt,self.leafNode)
                self.plotMidText((self.xOff,self.yOff),cntrPt,str(key))
        self.yOff=self.yOff+1/self.totalD
    
     #定义主函数，来调用其它函数   
    def createPlot(self,inTree):
        self.InitPram()
        fig=plt.figure(figsize=(10,5),facecolor='white')
        fig.clf()
        axprops=dict(xticks=[],yticks=[])
        self.ax1=plt.subplot(111,frameon=False,**axprops)
        #获取决策树叶子节点个数self.totalW
        self.totalW=float(self.getNumLeafs(inTree))
        #获取决策树深度self.totalD
        self.totalD=float(self.getTreeDepth(inTree))
        #self.xOff/self.yOff为最近绘制的一个叶节点的x/y坐标，
        self.xOff=-0.5/self.totalW;self.yOff=1.0;
        #传入决策树，首个根节点(这个根节点不具有实际意义)，根节点名为''(空值)
        self.plotTree(inTree,(0.5,1.1),'')
        plt.show()
