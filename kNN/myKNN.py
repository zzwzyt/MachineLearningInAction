# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:17:35 2017

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def kNN(X_train,y_train,X,k=3):
    diff = X_train-X
    dist = np.sqrt(np.sum(np.square(diff),axis=1))
    indexSorted = np.argsort(dist)       #argsort默认按从小到大取索引
    indexK = indexSorted[:k]
    labelK = y_train[indexK].flatten().tolist()
    classes = set(labelK)
#    #不使用字典
#    y_label = None
#    y_count = 0
#    for item in classes:
#        itemCount = labelK.count(item)
#        if itemCount > y_count:
#            y_count = itemCount
#            y_label = item
    
    classDict = {}
    #使用字典
#    自己实现的
#    for item in classes:
#        itemCount = labelK.count(item)
#        classDict[item] = itemCount
    #教材中的方法
    for i in range(k):
        voteIlabel = y_train[indexK[i]]
        classDict[voteIlabel] = classDict.get(voteIlabel,0) + 1
    classSorted = sorted(classDict.items(),key=lambda item:(item[1],item[0]),reverse=True)
    y_label = classSorted[0][0]
    
        
    return y_label

##测试kNN
#X = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
#labels = np.array([['A','A','B','B']]).flatten()
#X_test = np.array([[1,0.1]])
#y = kNN(X,labels,X_test)
#print(y)
    
##从文档加载数据加集
##可使用numpy.genfromtxt或者pandas.read_csv等
#data =  np.genfromtxt('datingTestSet2.txt',dtype=np.float32) 
#也可以使用open读取
def loadDataset(filename):
    dataList = []
    with open(filename) as f:
        for currLine in f.readlines():
            currLine = currLine.split('\t')
            currLine = [float(a) for a in currLine]
            dataList.append(currLine)
    return np.array(dataList)

if __name__ == '__main__':         
    dataArr = loadDataset('datingTestSet2.txt')  
    X = dataArr[:,:-1]
    y = dataArr[:,-1]
    
    #数据标准化
    X_min = np.min(X,axis=0,keepdims=True)
    X_max = np.max(X,axis=0,keepdims=True)
    X_stand = (X - X_min)/(X_max - X_min)
    
    #数据可视化
    pairs = [[0,1],[0,2],[1,2]]
    for i,pair in enumerate(pairs):
        subData = X_stand[:,pair]
        plt.figure(i)
        plt.scatter(subData[:,0],subData[:,1],c=y)
        plt.show()
    
    '''发现pair=[0,1]时分类效果较好'''
    X_new = X_stand[:,[0,1]]
    X_train,X_test,y_train,y_test = train_test_split(X_new,y,train_size=0.8)
    
    errorCount = 0
    for i,inX in enumerate(X_test):
        y_pred = kNN(X_train,y_train,inX,k=5)
        if y_pred != y_test[i]:
            print('Error predict:',y_pred,'The right answer:',y_test[i])
            errorCount += 1
    print('The error rate is ',errorCount/y_test.shape[0])
    
    



