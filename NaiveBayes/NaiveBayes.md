
# 朴素贝叶斯  
## 一、连续型  
当特征中数据是连续型时，通常有两种方法来估计条件概率。  
**第一种方法**是把每一个连续的数据离散化，然后用相应的离散区间替换连续数值。这种方法对于划分离散区间的粒度要求较高，不能太细，也不能太粗。  
**第二种方法**是假设连续数据服从某个概率分布，使用训练数据估计分布参数，通常我们用高斯分布来表示连续数据的类条件概率分布。  
此处我们使用第二种方法：  
### 1.计算高斯概率密度CalcuGaussProb(x,mean,stdev)  
公式为：
$\sqrt{\frac{1}{2πσ^2}}e^{(-\frac{1}{2σ^2}(x-μ)^2)}$  
μ为均值，$σ^2$为方差，σ为标准差  
给定来自训练数据中已知特征的均值和标准差后，可以使用高斯函数来评估一个给定的特征值的概率。即用高斯概率密度函数来表示条件概率$P(x^{(j)}|c_k)$


```python
#计算高斯概率密度函数
def CalcuGaussProb(self,x,mean,stdev):
    exponent = np.exp(-(np.power(x-mean,2))/(2*np.power(stdev,2)))
    GaussProb = (1/(np.sqrt(2*np.pi)*stdev))*exponent
    return GaussProb
```

### 2.获取各类别中各特征的均值、方差和标签集getMeanStdLabel(self,train_data)  
获取类标签，并获取每个类中各个特征的均值和方差


```python
#获取训练集每一类中每个特征的均值和方差以及类标签的取值集合
def getMeanStdLabel(self,train_data):
    label_counts=train_data.label.value_counts()
    label_arr=np.array(label_counts.index)
    label_arr.sort()
    #得到除标签外特征数
    num_feature = len(train_data.columns.values) - 1
    #按类别划分数据
    names = locals()
    for i in range(len(label_arr)):
        names['c%s' % i] = train_data[train_data["label"]==label_arr[i]]
    #按类别对每个属性求均值和方差
    c_mean=[]
    c_std=[]
    for j in range(len(label_arr)):
        names['mc%s' % j] = []
        names['sc%s' % j] = []
        for k in range(num_feature):
            names['mc%s' % j].append(np.mean(names['c%s' % j][k]))
            names['sc%s' % j].append(np.std(names['c%s' % j][k],ddof=1))

    for x in range(len(label_arr)):
        c_mean.append(names['mc%s' % x])
        c_std.append(names['sc%s' % x])
        names['arr_c%s' % x] = np.array(names['c%s' % x])
    return c_mean,c_std,label_arr
```

### 3.  计算连续型数据所属类的概率CalcuClassProbCon(arr,cx_mean,cx_std)
n个样本的样本集为$x_i\in\{x_1,x_2,...,x_n\}$，第i个样本$x_i$有m个特征$x_i^{(j)}\in\{x_i^{(1)},x_i^{(2)},...,x_i^{(m)}\}$  
对于**一个样本属于某类的概率**，我们用这个样本所有特征概率之乘积来表示，即$\prod\limits_{j=1}^mP(x^{(j)}|c_k)$  


```python
#计算连续数据所属类的概率
def CalcuClassProbCon(self,arr,cx_mean,cx_std):
    cx_probabilities=1
    for i in range(len(cx_mean)):
        cx_probabilities *= self.CalcuGaussProb(arr[i],cx_mean[i],cx_std[i])
    return cx_probabilities
```

### 4.获取单个样本的预测类别predict(arr,cmean,cstd,label_array)
对于单个样本返回预测结果，即比较所有类别下，这个样本的概率，找到最大的概率值，返回其类别和概率值。  
传入测试样本、均值、方差和标签集合。


```python
#单一样本预测
def predict(self,trainData,testData):
    prob = []
    #print(trainData)
    self.cmean,self.cstd,self.label_array=self.getMeanStdLabel(trainData)
    for i in range(len(self.cmean)):
        cx_mean = self.cmean[i] #x类的均值
        cx_std = self.cstd[i] #x类的方差
        #print(testData)
        prob.append(self.CalcuClassProbCon(testData,cx_mean,cx_std)) #将计算得到的各类别概率存入列表
    bestLabel,bestProb = None,-1 #初始化最可能的类和最大概率值    
    for i in range(len(prob)): #找到所有类别中概率值最大的类
        if prob[i] > bestProb:
            bestProb = prob[i]
            bestLabel = self.label_array[i]
    return bestLabel,bestProb
```

### 5.获取整个数据集的预测结果getPredictions(testarr,cmean,cstd,label_array)


```python
#整个数据集预测
def getPredictions(self,TrainData,TestData):
    self.prediction = []
    self.testdata = np.array(TestData)
    for i in range(len(self.testdata)):
        result,proby = self.predict(TrainData,self.testdata[i])
        self.prediction.append(result)
    return self.prediction
```

### 6.计算准确率
同理推广到整个数据集(测试集)上，通过比对预测结果和真实标签，计算出准确率  
$acc=\frac{预测正确数}{数据集总数}$


```python
#计算准确性
def getAccuracy(self):
    correct = 0
    for i in range(len(self.testdata)):
        if(self.testdata[i][-1]==self.prediction[i]):
            correct += 1
    return (correct/float(len(self.testdata)))*100.0
```

## 二、离散型  
### 1.初始化__init__(lamda=1)  
lamda为贝叶斯平滑因子，默认取1(即拉普拉斯平滑)


```python
def __init__(self,lamda=1):
    self.lamda = lamda
```

### 2.获取相关参数 getParams(data)  


```python
#获取相关参数
def getParams(self,data):
    self.ck_counts = data.label.value_counts()#训练样本中类为ck的数量集合
    self.ck_name = np.array(self.ck_counts.index)#训练样本中类ck名称集合    
    self.DataNum = len(data)#训练样本总数N
    self.CNum = len(self.ck_counts)#类的个数K
    self.DataSet = data
```

### 3.计算先验概率 calPriorProb()  
先验概率：$P_λ(Y=c_k)=\frac{\sum\limits_{i=1}^NI(y_i=c_k)+λ}{N+Kλ}$  
$\sum\limits_{i=1}^NI(y_i=c_k)$，用**ck_counts**表示


```python
#计算先验概率
def calPriorProb(self):
    self.ck_PriorProb = []
    for i in range(self.CNum):
        cx_PriorProb = (self.ck_counts[i]+self.lamda)/(self.DataNum+self.CNum*self.lamda)
        self.ck_PriorProb.append(cx_PriorProb)
```

### 4.计算条件概率 calCondProb()  
条件概率：$P_λ(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum\limits_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)+λ}{\sum\limits_{i=1}^NI(y_i=c_k)+mλ}$  
我们将计算得到的每个类的每个特征取值的条件概率存起来


```python
#计算条件概率
def calCondProb(self):
    names = locals()#使用动态变量
    self.CondProb = []#存储所有类别的所有特征取值的条件概率
    self.feat_value = []#所有特征取值列表

    #对于每一类别的数据集
    for i in range(len(self.ck_name)):
        names['Q%s' % i] = self.DataSet[self.DataSet["label"]==self.ck_name[i]]#按类别划分数据集
        names['ConProbC%s' % i] = []#定义动态变量，表示各类别中所有特征取值的条件概率集合
        feature_arr = self.DataSet.columns.tolist()[0:len(self.DataSet.columns)-1]#获取训练数据集特征集

        #对于每一个特征求该特征各个取值的条件概率
        for feature in (feature_arr):

            names['Q%s' % feature]=[]#定义动态变量，表示某个类别的某个特征的所有取值条件概率

            #对于某个特征的所有可能取值求条件概率
            for value in self.DataSet[feature].value_counts().index.tolist():

                #生成所有特征取值列表
                if value not in self.feat_value:#如果这个取值不在列表中，则加入这个取值
                    self.feat_value.append(value)

                #这里用了拉普拉斯平滑，使得条件概率不会出现0的情况
                #如果某个类的某个特征取值在训练集上都出现过，则这样计算
                if value in names['Q%s' % i][feature].value_counts():
                    temp = (names['Q%s' % i][feature].value_counts()[value]+self.lamda)/(names['Q%s' % i][feature].value_counts().sum()+len(names['Q%s' % i][feature].value_counts())*self.lamda)
                #如果某个类的某个特征取值并未在训练集上出现，为了避免出现0的情况，分子取1(即lamda平滑因子，取1时为拉普拉斯平滑)
                else:
                    temp = self.lamda/(names['Q%s' % i][feature].value_counts().sum()+len(names['Q%s' % i][feature].value_counts())*self.lamda)

                #将求得的特征取值条件概率加入列表
                names['Q%s' % feature].append(temp)
            #将得到的某个类别的某个特征的所有取值条件概率列表加入某个类别中所有特征取值的条件概率集合
            names['ConProbC%s' % i].extend(names['Q%s' % feature])
        #将某个类别中所有特征取值的条件概率集合加入所有类别所有特征取值的条件概率集合
        self.CondProb.append(names['ConProbC%s' % i])
    #将所有特征取值列表也加入所有类别所有特征取值的条件概率集合(后面用来做columns--列索引)
    self.CondProb.append(self.feat_value)
    #用类别名称的集合来生成行索引index
    index = self.ck_name.tolist()
    index.extend(['other'])#此处由于我最后一行是feat_value，后面会删掉，因此在行索引上也多加一个，后面删掉
    #将所有类别所有特征取值的条件概率集合转换为DataFrame格式
    self.CondProb = pd.DataFrame(self.CondProb,columns=self.CondProb[self.CNum],index = index)
    self.CondProb.drop(['other'],inplace = True)
```

### 5.预测给定一个实例 predict(traindata,testdata)


```python
#对一个样本进行预测    
def predict(self,traindata,testdata):
    self.getParams(traindata)#获取参数
    self.calPriorProb()#获取先验概率
    self.calCondProb()#获取条件概率

    self.ClassTotalProb = []#初始化各类别总概率列表
    bestprob = -1#初始化最高概率
    bestfeat = ''#初始化最可能类别

    for feat in self.ck_name:
        pp = self.ck_PriorProb[self.ck_name.tolist().index(feat)]#pp为先验概率
        cp = 1#初始化条件概率
        for value in self.feat_value:
            if value in testdata.value_counts().index.tolist():
                cp = cp * self.CondProb[value][feat]#计算各特征取值的条件概率之积
        TotalProb = pp * cp#条件概率之积与先验概率相乘
        self.ClassTotalProb.append(TotalProb)
    #找到最可能类别和其概率    
    for i in range(len(self.ck_name)):
        if self.ClassTotalProb[i] > bestprob:
            bestprob = self.ClassTotalProb[i]
            bestfeat = self.ck_name[i]
    return bestprob,bestfeat
```

### 6.计算预测准确度getAccuracy(traindata,testdata)  


```python
#计算预测准确度
def getAccuracy(self,traindata,testdata):
    num = 0
    realFeat = testdata.label.tolist()
    for i in range(len(testdata)):
        temp = testdata.iloc[i][0:len(testdata.columns)-1]    
        predProb,predFeat = self.predict(traindata,temp)
        print(predProb,predFeat,realFeat[i])
        if(realFeat[i] == predFeat):
            num = num + 1
    acc = num / len(realFeat)
    return acc
```

## 三、数据集测试  
### 1.连续型  
用的是pima印第安人糖尿病数据集来测试


```python
diabetes = pd.read_csv(path+"pima-indians-diabetes.csv")
dia_train,dia_test = train_test_split(diabetes,test_size=0.1)
model_NBC = NaiveBayesContinuous()
model_NBC.getPredictions(dia_train,dia_test)
acc1 = model_NBC.getAccuracy()
print("准确率：","%.2f" % acc1,"%")
```

    准确率： 83.12 %
    

### 2.离散型  
用的是汽车性价比CarEvalution数据集来测试


```python
car = pd.read_csv(path+"CarEvalution.csv")
car_train,car_test = train_test_split(car,test_size=0.1)
model_NBD = NaiveBayesDiscrete()
acc2 = model_NBD.getAccuracy(car_train,car_test)
print("%.2f" % acc2,"%")
```

    87.02 %
    
