
## 支持向量机 
### 1.初始化 \__init\__(C=1,toler=0,kernelInfo=('linear',0),n_iter=100)  
默认C=1(C为松弛变量),toler=0(toler为容错率),核函数是线性核，迭代次数100次


```python
def __init__(self,C=1,toler=0,kernelInfo=('linear',0),n_iter=100):
    self.C = C
    self.toler = toler
    self.kernelInfo = kernelInfo
    self.n_iter = n_iter
```

### 2.计算核函数数值 calKernelValues(data,datax)  
这里写了两种核函数，其他核函数可以后续加上  
**线性核** $K(x,z)=x^Tz$  
**高斯核(一种径向基核)** $K(x,z)=e^{-\frac{||x-z||^2}{2δ^2}}$


```python
#计算核函数
def calKernelValues(self,data,datax):
    datalen = len(data)
    k_mat = np.zeros((datalen,1))#shape:(datalen,1)
    if self.kernelInfo[0] == 'linear':
        #线性核函数 K(x,z) = xz
        #data shape:(datalen,featnum)  datax.T shape :(featnum,)
        #(datalen,featnum) × (featnum,) --> (datalen,)
        k_mat = np.dot(data,datax.T)

    elif self.kernelInfo[0] == 'rbf':
        #径向基核函数 K(x,z) = exp((||x-z||^2)/(-2*delta^2))
        #循环后 --> [datalen×1]
        for j in range(datalen):

            deltaRow = data[j,:] - datax
            #[1×featnum] × [featnum×1] --> [1×1]
            #求出||x-z||^2 
            k_mat[j] = np.dot(deltaRow,deltaRow.T) 

        #[datalen×1] / [1×1] --> [datalen×1]
        #求出exp((||x-z||^2)/(-2*delta^2))
        k_mat = np.exp(k_mat/(-1*self.kernelInfo[1]**2)).reshape(datalen)

    else:
        raise NameError('That Kernel is not recognized!')

    return k_mat
```

### 3.计算$α_k$的差值$E_k$ calErrorValues(k)  
$E_k=g(x_k)-y_k=\sum\limits_{j=1}^mα_j^*y_jK(x_k,x_j)+b-y_k$  
此处过程：  
1.计算$α_j^*y_j$  
2.获取$K(x_k,x_j)$  
3.求得$g(x_k)$  
4.求得$E_k$


```python
#对于给定的alphak计算差值Ek
def calErrorValues(self,k):
    #计算alphaj * yj
    #[datanum×1] 对应相乘后转置 --> [1×datanum]
    alphay = np.multiply(self.alphas.T,self.label)

    #获取Kjk = K(xk,xj)，第k列
    #[datanum × 1] 
    Kjk = self.K[:,k]

    #g(xk) = sum(alphaj * yj) × K(xj,xk) + b
    #[1×datanum] × [datanum×1] + [1×1] --> [1×1]
    gxk = float(np.dot(alphay,Kjk) + self.b)

    #Ek = g(xk) - yk
    #[1×1] - [1×1] --> [1×1]
    Ek = gxk - float(self.label[k])

    return Ek       
```

### 4.存储差值$E_k$ updateErrorCache(k)  
将计算得到的$E_k$存入差值缓存ErrorCache，实际存入的是一个列表，第0位的1表示这个差值是有效的，第1位存放差值。


```python
#存储差值
def updateErrorCache(self,k):
    #计算第k个alpha的差值
    error = self.calErrorValues(k)
    #存入ErrorCacha，第一位的1表示这个差值是否有效
    self.ErrorCache[k] = [1,error]
```

### 5.选择第二个变量$α_j$和差值$E_j$  selectAlphaj(i,Ei)  
传入第一个变量的下标i和差值$E_i$,返回第二个变量的下标j和差值$E_j$  
    若是有效差值列表中有有效差值找出使$△E=E_i-E_j$最大的$E_j$，若没有则随机选取一个不为$E_i$的$E_j$


```python
#选择第二个变量alphaj的差值Ej(传入第一个变量Ei和下标i)
def selectAlphaj(self,i,Ei):
    #用maxk和maxEk表示使步长|Ei-Ej|最大的alphak的下标和Ek
    maxk,maxEk = -1,0
    #初始化Ej
    Ej = 0
    #将Ei在cache中设置为有效
    self.ErrorCache[i] = [1,Ei]
    #有效差值列表，用nonzero获取不为0的值的下标
    vaildErrorCacheList = np.nonzero(self.ErrorCache[:,0])[0]

    #若是有效差值列表中有有效差值(条件>1是因为该表中必然有一个有效差值,即Ei)
    if len(vaildErrorCacheList) > 1:
        #遍历有效差值列表
        for k in vaildErrorCacheList:
            #若找到Ei则跳过继续下一个
            if k == i: 
                continue
            #计算Ek
            Ek = self.calErrorValues(k)                
            #计算△E=|Ei-Ek|
            deltaE = np.abs(Ei-Ek)
            if deltaE > maxEk:
                maxk = k
                maxEk = deltaE
            #找到最终使|Ei-Ek|最大的Ek赋值给Ej
            Ej = Ek
        return maxk,Ej

    #若是有效差值列表中没有有效差值(排除Ei)
    else:
        #在范围内随机选取一个j，计算Ej
        j = self.selectRandj(i)
        Ej = self.calErrorValues(j)

    return j,Ej 
```

#### 辅助函数：随机选取下标不为i的j selectRandj(i)


```python
#随机选取下标不为i的j
def selectRandj(self,i):
    j = i
    while(j==i):
        j = int(random.uniform(0,self.datanum))
    return j
```

### 6.内循环 InnerLoop(i)  
传入第一个变量的下标i，返回1表示$α_j$被修改，返回0表示$α_j$未被修改  
此处过程：  
1.计算第一个变量$α_i$的差值$E_i$;  
2.判断是否违背KKT约束条件，此处使用辅助函数violateKKT(Ei,i);  
3.在约束条件下求最小;  
若$y_i\neq y_j\to α_i-α_j=k$则$L=max(0,α_j-α_i)$，$H=min(C,C+α_j-α_i)$  
若$y_i = y_j\to α_i+α_j=k$则$L=max(0,α_j+α_i-C)$，$H=min(C,α_j+α_i)$   
若是求得的L=H则不改变$α_i$  
4.计算η  
$η= 2K_{ij}-K_{ii}+K_{jj}$  
5.计算沿着约束方向未经剪辑的解$α_j^{new.unc}=α_j^{old}+\frac{y_j(E_i-E_j)}{η}$，并调整$α_j$，此处使用辅助函数clipAlpha(aj,H,L)  
6.存储差值$E_j$  
7.若$α_j$和$α_j^{old}$改变值的绝对值小于0.000001，则认定$α_j$未修改  
8.由经剪辑后的$α_j$求$α_i$，计算并存储$E_i$  
9.计算阈值b，此处使用辅助函数getb(Ei,Ej,i,j,alphai_old,alphaj_old)


```python
#内循环
def InnerLoop(self,i):
    #1.计算第一个变量alphai的差值Ei
    Ei = self.calErrorValues(i)
    
    #2.选择违背KKT约束条件的alphai
    if self.violateKKT(Ei,i):
        j,Ej = self.selectAlphaj(i,Ei)
        alphai_old = self.alphas[i].copy()
        alphaj_old = self.alphas[j].copy()
        
        #3.计算L和H
        if self.label[i] != self.label[j]:
            L = np.maximum(0,(self.alphas[j]-self.alphas[i])[0])
            H = np.minimum(self.C,self.C+self.alphas[j]-self.alphas[i])
        else:
            L = np.maximum(0,self.alphas[j]+self.alphas[i]-self.C)
            H = np.minimum(self.C,self.alphas[j]+self.alphas[i])

        if L==H:
            return 0
        
        #4.计算η
        Kij = np.dot(self.TrainData[i,:],self.TrainData[j,:].T)
        Kii = np.dot(self.TrainData[i,:],self.TrainData[i,:].T)
        Kjj = np.dot(self.TrainData[j,:],self.TrainData[j,:].T)
        eta = 2.0 * Kij - Kii - Kjj            
        if eta >= 0:
            return 0
        
        #5计算αj^{new,unc}
        self.alphas[j] -= self.label[j] * (Ei-Ej) / eta
        #print("调整前aj:",self.alphas[j])
        self.alphas[j] = self.clipAlpha(self.alphas[j],H,L)
        #print("调整后aj:",self.alphas[j])
        
        #6.存储Ej
        self.updateErrorCache(j)
        
        #7.若aj和aj^{old}改变值的绝对值小于0.000001，则认定αj未修改
        if(np.abs(self.alphas[j]-alphaj_old)<0.000001):
            return 0
        
        #8.由经剪辑后的α_j求α_i
        self.alphas[i] = self.alphas[i] + self.label[j]*self.label[i]*(alphaj_old-self.alphas[j])
        #计算并存储Ei
        self.updateErrorCache(i)
        #9.计算阈值
        self.getb(Ei,Ej,i,j,alphai_old,alphaj_old)

        return 1
    else:
        return 0
```

**辅助函数:计算是否违背KKT约束**  


```python
def violateKKT(self,Ei,i):
    if (self.label[i]*Ei<-1*self.toler)&(self.alphas[i]<self.C):
        return True
    elif (self.label[i]*Ei>self.toler)&(self.alphas[i]>0):
        return True
    else:
        return False
```

**辅助函数:计算阈值b**  
1.先求出$y_iK_{ii}α_i、y_jK_{ji}α_j、y_iK_{ij}α_i、y_jK_{jj}α_j$，其中$α_i=α_i^{new}-α_i^{old}$，$α_j=α_j^{new}-α_j^{old}$  
    
2.求出$b_i$和$b_j$：      
　　若$α_i>0$且$α_i<C$  
　　$b_i^{new}=-E_i-y_iK_{ii}α_i-y_jK_{ji}α_j$  
　　若$α_j>0$且$α_j<C$  
　　$b_j^{new}=-E_j-y_iK_{ij}α_i-y_jK_{jj}α_j$  
    
3.求得$b^{new}=\frac{b_i^{new}+b_j^{new}}{2}$


```python
def getb(self,Ei,Ej,i,j,alphai_old,alphaj_old):
    yiKii_alphai = self.label[i]*np.dot(self.TrainData[i,:],self.TrainData[i,:].T)*(self.alphas[i]-alphai_old)
    yjKji_alphaj = self.label[j]*np.dot(self.TrainData[j,:],self.TrainData[i,:].T)*(self.alphas[j]-alphaj_old)
    yiKij_alphai = self.label[i]*np.dot(self.TrainData[i,:],self.TrainData[j,:].T)*(self.alphas[i]-alphai_old)
    yjKjj_alphaj = self.label[j]*np.dot(self.TrainData[j,:],self.TrainData[j,:].T)*(self.alphas[j]-alphaj_old)
    bi = self.b - Ei - yiKii_alphai - yjKji_alphaj
    bj = self.b - Ej - yiKij_alphai - yjKjj_alphaj
    if (self.alphas[i]>0) & (self.alphas[i]<self.C):
        self.b = bi
    elif (self.alphas[j]>0) & (self.alphas[j]<self.C):
        self.b = bj
    else:
        self.b = (bi+bj)/2.0
```

### 7.训练SVM(外循环)


```python
def fit(self,TrainData):
    self.datanum = len(TrainData) #训练样本数量
    self.featnum = len(TrainData.columns) - 1 #特征数量
    self.TrainData =  np.array(TrainData[TrainData.columns.tolist()[0:self.featnum]]) #训练数据集
    self.label = np.array(TrainData.label) #训练数据集标签
    self.alphas = np.zeros((self.datanum,1))
    self.b = 0
    self.ErrorCache = np.zeros((self.datanum,2))
    self.K = np.zeros((self.datanum,self.datanum))
    for i in range(self.datanum):
        self.K[:,i] = self.calKernelValues(self.TrainData,self.TrainData[i,:]) #self.K[:,i] shape:(detanum)
    x_iter = 0
    entireSet = True
    alphaPairsChanged = 0 #遍历整个数据集修改任意alpha的次数
    #若
    while(x_iter<self.n_iter)&((alphaPairsChanged>0)|(entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(self.datanum):
                #若alphai被修改，则次数+1
                alphaPairsChanged += self.InnerLoop(i)
                #print("fullset,iter: %d i:%d,pairs changed %d" % (x_iter,i,alphaPairsChanged))
            x_iter += 1
        else:
            #获取0<alpha<C的坐标
            nonBoundIs = np.nonzero((self.alphas>0)&(self.alphas<self.C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += self.InnerLoop(i)
                #print("non-bound,iter: %d i:%d,pairs changed %d" % (x_iter,i,alphaPairsChanged))
            x_iter += 1
        print("iter:",x_iter)
        print("修改次数",alphaPairsChanged)
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

    return self.b,self.alphas
```

### 8.计算准确率、精确率、召回率和F1score


```python
def getAccuracy(self,TestData):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    num = len(TestData)
    test_label = np.array(TestData.label)
    test_data =  np.array(TestData[TestData.columns.tolist()[0:self.featnum]])
    supportVectorsIndex = np.nonzero(self.alphas>0)[0]
    supportVectors = self.TrainData[supportVectorsIndex]
    supportVectorLabels = self.label[supportVectorsIndex]
    supportVectorAlphas = self.alphas[supportVectorsIndex]
    for i in range(num):  
        kernelValue = self.calKernelValues(supportVectors, test_data[i,:])  
        predict = np.dot(kernelValue.T,np.multiply(supportVectorLabels, supportVectorAlphas.T)[0]) + self.b 
        if ((int(np.sign(predict))>0) & (np.sign(test_label[i])>0)):             
            TP = TP + 1 
        elif ((int(np.sign(predict))<0) & (np.sign(test_label[i])<0)):
            TN = TN + 1
        elif ((int(np.sign(predict))>0) & (np.sign(test_label[i])<0)):
            FP = FP + 1
        elif ((int(np.sign(predict))<0) & (np.sign(test_label[i])>0)):
            FN = FN + 1
    accuracy = (TP+TN) / (TP+FP+TN+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F1_score = 2*recall*precision/(recall+precision)
    evaluate = [accuracy,precision,recall,F1_score]
    return  evaluate
```
