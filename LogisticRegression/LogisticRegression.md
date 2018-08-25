
## 逻辑回归  
### 1.初始化__init__(n_iter=100,eta=0.0001,gd='bgd')  
初始化逻辑回归类，默认迭代次数为100，学习率默认为0.0001，训练默认为批量梯度下降，默认不绘制决策边界，默认用L2正则化(L1正则化只写了损失函数，训练没有写)。  
训练可以选择随机梯度下降，可以选择打印决策边界(对于多特征数据，只显示前两个特征构成的平面)。


```python
#初始化
def __init__(self,n_iter = 100,eta = 0.0001,gd = 'bgd',plot = False,regular = 'none',lamda = 1,showcost = False):
    self.n_iter = n_iter #迭代次数
    self.eta = eta #学习率
    self.gd = gd #学习方式
    self.plot = plot #绘制决策边界变化
    self.regular = regular #正则化
    self.lamda = lamda #正则化惩罚系数
    self.showcost = showcost #显示迭代次数和cost值
```

### 2.对数函数sigmoid(fx)  
通过对数几率函数将预测值范围限定在0到1之间，并且该函数单调可微


```python
#对数几率函数    
def sigmoid(self,fx):
    return 1/(1+np.exp(-1 * fx))
```

### 3.训练fit(TrainData)  
损失函数为：$L(w)=-\frac{1}{m}\sum\limits_{i=1}^m(y_iln[h(x_i)]+(1-y_i)ln[1-h(x_i)])$   
求偏导得：$\frac{\partial L(w)}{\partial w}=-\frac{1}{m}\sum\limits_{i=1}^m[y_i-h(x_i)]x_i$  
更新权重：$w = w - α\frac{1}{m}\sum\limits_{i=1}^m[h(x_i)-y_i]x$  
(训练方式可选择批量梯度下降或是随机梯度下降，可以通过初始化类时的plot参数选择是否显示决策边界变化情况)  
加上L2正则化后$E(w)=L(w)+λ\sum\limits_{j=1}^nw_j^2$  
求导$\frac{\partial L(w)}{\partial w}=-\frac{1}{m}\sum\limits_{i=1}^m[y_i-h(x_i)]x_i+2λ\sum\limits_{j=1}^nw_j$  


```python
#训练
def fit(self,TrainData):

    self.datanum = len(TrainData)#获取训练集大小
    self.featnum = len(TrainData.columns)-1#获取特征数量        
    self.label = np.array(TrainData.label)#获取真实标签集合
    self.data = np.array(TrainData[TrainData.columns.tolist()[0:self.featnum]])#datanum×featnum
    self.data_T = self.data.transpose()#数据大小为featnum×datanum
    self.weights = np.zeros((1,self.featnum))#初始化权重，权重大小为1×featnum                
    self.warr = []
    #批量梯度下降
```

#### 批量梯度下降


```python
    if self.gd == 'bgd':
        for n in range(self.n_iter):
            #sigmoid([1×featnum] × [featnum×datanum]) --> [1×datanum]
            #预测值，即h(x)=1/(1+e^(-wx))
            hx = self.sigmoid(np.dot(self.weights,self.data_T)) 

            #[1×datanum] - [1×datanum] --> [1×datanum]
            #h(x)-y,误差值
            loss = hx - self.y_label 

            #L2正则化项求导,更新每个权重时都要加上，shape为[1×featnum]
            penalty = np.ones((1,self.featnum))*2*self.lamda*self.weights.sum()

            if self.regular == 'l2':
                #([1×datanum] × [datanum×featnum] + [1×featnum])/datanum --> [1×featnum]
                #([h(x)-y]x + 2*lamda*w)/m
                gradient = (np.dot(loss,self.data) + penalty) / self.datanum 

            elif self.regular == 'none': 
                #([1×datanum] × [datanum×featnum])/datanum --> [1×featnum]
                #[h(x)-y]x/m
                gradient = (np.dot(loss,self.data)) / self.datanum

            #[1×featnum] - eta*[1×featnum] --> [1×featnum]
            self.weights = self.weights - self.eta * gradient

            #显示迭代次数和损失值
            if (n % 100 == 0) & self.showcost:
                self.cost = self.costFunction()
                print("迭代",n,"次","损失值为：",self.cost)

            #可以选择训练过程中绘制决策边界
            if self.plot == True:
                if n % (self.n_iter/5) == 0:
                    self.plotDecisionBoundary(TrainData)
        if self.plot == True:
            self.plotDecisionBoundary(TrainData)

        return self.weights
```

#### 随机梯度下降


```python
    #随机梯度下降
    if self.gd == 'sgd':
        for n in range(self.n_iter):
            #生成一个随机数x
            x = random.randint(0,self.datanum-1)  
            datax = self.data[x]  #[1×featnum]
            datax_T = datax.transpose()  #[featnum×1]

            #sigmoid([1×featnum] × [featnum×1]) --> [1×1]
            hxx = self.sigmoid(np.dot(self.weights,datax_T))

            #[1×1] - [1×1] --> [1×1]
            lossx = hxx-self.label[x]

            #L2正则化项求导,更新每个权重时都要加上，shape为[1×featnum]
            penalty = np.ones((1,self.featnum))*2*self.lamda*self.weights.sum() 


            if self.regular == 'l2':
                #([1×1] × [1×featnum] + [1×featnum]) --> [1×featnum]
                gradientx = (lossx * datax) + penalty
            if self.regular == 'none':
                #([1×1] × [1×featnum]) --> [1×featnum]
                gradientx = lossx * datax


            #[1×featnum] - eta*[1×featnum] --> [1×featnum]
            self.weights = self.weights - self.eta * gradientx
            self.warr.append(self.weights[0].tolist())
            #当损失值小于0.1时停止迭代
            if self.costFunction() < 0.1:
                break
            #显示迭代次数和损失值
            if (n % 100 == 0) & self.showcost:
                self.cost = self.costFunction()
                print("迭代",n,"次","损失值为：",self.cost)

            #可以选择训练过程中绘制决策边界
            if self.plot == True:
                if n % (self.n_iter/5) == 0:
                    self.plotDecisionBoundary(TrainData)
        if self.plot == True:
            self.plotDecisionBoundary(TrainData)

        return self.weights
```

### 4.计算损失值costFunction()


```python
#计算损失值
def costFunction(self):
    #sigmoid([1×featnum] × [featnum×datanum]) --> [1×datanum]
    h = self.sigmoid(np.dot(self.weights,self.data_T))

    #计算损失函数:E(w)=L(w)+lamda*R(w)
    #E(w) = -1/datanum * [y*ln(h(x)) + (1-y)*ln(1-h(x))]

    #R(w) = ||w||1 权重绝对值之和
    if self.regular == 'l1':
        #lamda*1 --> [1×1]
        C = self.lamda * np.abs(self.weights).sum()#L1正则化项

        #([1×datanum]×[datanum×1]) + ([1×datanum]×[datanum×1]) --> [1×1]
        cost = (-1/self.datanum) * (np.dot(np.log(h),self.label)+(np.dot(np.log(1-h),1-self.label)) + C)

    #R(w) = ||w||2 权重平方之和   
    elif self.regular == 'l2':
        #lamda * ([1×featnum] × [featnum×1]) --> [1×1]
        C = self.lamda * np.dot(self.weights,self.weights.transpose())#L2正则化项
        #([1×datanum]×[datanum×1]) + ([1×datanum]×[datanum×1]) --> [1×1]
        cost = (-1/self.datanum) * (np.dot(np.log(h),self.label)+(np.dot(np.log(1-h),1-self.label)) + C)            


    elif self.regular == 'none':
        cost = (-1/self.datanum) * (np.dot(np.log(h),self.label)+(np.dot(np.log(1-h),1-self.label)))

    return cost
```

### 5.预测单个样本类别predict(testData)  
当$w·x > 0$时为1，当$w·x < 0$时为0


```python
#预测单一样本
def predict(self,testData):
    flag = np.dot(self.weights,testData)
    if flag > 0:
        pred = 1
    else:
        pred = 0
    return pred
```

### 6.求准确率accuracy(TestData)  


```python
#获取数据集准确率
def accuracy(self,TestData):
    num = 0
    for i in range(len(TestData)):
        temp = np.array(TestData.iloc[i][0:len(TestData.columns)-1]).reshape(len(TestData.columns)-1,1)
        if self.predict(temp)==TestData.label.tolist()[i]:
            num = num + 1
    return num/len(TestData)
```

### 7.绘制决策边界plotDecisionBoundary(TrainData)


```python
def plotDecisionBoundary(self,TrainData):
    fig=plt.figure(figsize=(10,8))
    plt.xlim(-4,4)  #  设置x轴刻度范围
    plt.ylim(-4,4)  #  设置y轴刻度范围
    plt.xlabel('x1')   
    plt.ylabel('x2')
    plt.title('decision boundary') 
    x1 = np.arange(-4,4,1)
    x2 =-1 * model_lr.weights[0][0] / model_lr.weights[0][1] * x1
    plt.scatter(TrainData[TrainData.columns[0]], TrainData[TrainData.columns[1]], c=TrainData['label'], s=30)
    plt.plot(x1,x2)
    plt.show()
```

## 绘制损失函数等值线和三维图  
对于只有两个特征的数据，我们可以可视化其权重与损失值的等值线图和三维图  
### 1.根据迭代得到的权重集合计算相应损失值calCost(data,warr)


```python
def calCost(self,data,warr):
        datanum = len(data)
        featnum = len(data.columns)-1
        label = np.array(data.label)#获取真实标签集合
        data = np.array(data[data.columns.tolist()[0:featnum]])
        self.w1 = []
        self.w2 = []
        for i in range(len(warr)):
            self.w1.append(warr[i][0])
            self.w2.append(warr[i][1])
        #为了防止迭代次数过多导致权重矩阵过多影响运算速度，只取前1000个权重
        if(len(warr) > 1000):
            self.w1 = self.w1[0:1000]
            self.w2 = self.w2[0:1000]
            
        self.xd = np.array(self.w1).min()#获取x下限
        self.xu = np.array(self.w1).max()#获取x上限
        self.yd = np.array(self.w2).min()#获取y下限
        self.yu = np.array(self.w2).max()#获取y上限
        #将w1，w2转为网格数据
        self.w1,self.w2 = np.meshgrid(np.array(self.w1),np.array(self.w2))
        self.cost = 0
        for i in range(datanum):
            fx = self.w1 * data[i][0]+ self.w2 * data[i][1]
            hx = 1/(1+np.exp(-1 * fx))
            self.cost = self.cost + (label[i]*np.log(hx)+(1-label[i])*np.log(1-hx))/datanum
            if i % 100 == 0:
                print("迭代",i,"次","一共",datanum,"次","  请等待")
        self.regular_cost = np.abs(self.w1)+np.abs(self.w2)
```

### 2.绘制等值线图plotContour()


```python
def plotContour(self):
        #self.calCost(data,warr)
        plt.figure(figsize=(8,8))
        plt.xlim(self.xd,self.xu)
        plt.ylim(self.yd,self.yu)
        #绘制原始损失函数L(w)
        P=plt.contour(self.w1,self.w2,self.cost,10)
        plt.clabel(P, alpha=0.75, cmap='jet',inline=1, fontsize=10)
        #绘制正则化损失函数R(w)
        C=plt.contour(self.w1,self.w2,self.regular_cost,1)
        plt.clabel(C, alpha=0.75, cmap='jet',inline=1, fontsize=10)
        plt.show()
```

### 3.绘制三维图plot3D()


```python
def plot3D(self):
        #self.calCost(data,warr)
        fig = plt.figure()
        ax =fig.add_subplot(111,projection='3d')
        ax.plot_surface(self.w1,self.w2,self.cost,rstride=3,cstride=3,cmap=cm.jet)
        plt.show()
```

**逻辑回归还有很多可以添加的，包括L1正则化，亦或是衰减学习率等，这些后续会逐步加上**
