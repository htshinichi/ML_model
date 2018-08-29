import numpy as np
class LinearRegression():
    def __init__(self,n_iter=100,gd='bgd',regular='none'):
        self.n_iter = n_iter
        self.gd = gd
        self.regular = regular
    def fit(self,TrainData):
        featnum = len(TrainData.columns)-1
        datanum = len(TrainData)
        self.train_data = np.array(TrainData[TrainData.columns.tolist()[0:featnum]])
        self.train_label = np.array(TrainData.label)
        #self.weights = np.ones((1,featnum))
        #self.bias = 0
        train_data_mean = np.tile(np.mean(self.train_data,axis=0),(datanum,1))
        fz = np.dot(self.train_label.T,self.train_data)-np.dot(self.train_label.T,train_data_mean)
        fm = np.sum(self.train_data**2,axis=0) - np.sum(self.train_data,axis=0)**2/datanum
        self.weights = fz / fm
        self.bias = (np.sum(self.train_label) - np.sum(np.dot(self.weights,self.train_data.T)))/datanum
        
    def predict(self,x_sample):
        prediction = np.dot(self.weights,x_sample.T)+self.bias
        return prediction
