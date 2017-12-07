# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:20:26 2017

@author: User
"""

import numpy as np
import logRegres

#示例：从疝气病症预测马的死亡率

#加载数据
def loadDataset(filename):
    dataArr = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineList = line.strip().split('\t')
            lineList = [float(i) for i in lineList]
            a = [1.0] #添加偏置项bias
            a.extend(lineList)
            dataArr.append(a)
    return np.array(dataArr)
#data = np.genfromtxt('horseColicTraining.txt')

def colicTest():
    trainData = loadDataset('horseColicTraining.txt')
    testData = loadDataset('horseColicTest.txt')
    X_train = trainData[:,:-1]
    y_train = trainData[:,-1]
    X_test = testData[:,:-1]
    y_test = testData[:,-1]
    
    weights,costs = logRegres.stocGradAscent1(X_train,y_train)
    y_pred = logRegres.predict(X_test,weights)
    
    errorCount = 0
    for i in range(y_test.shape[0]):
        if y_pred[i] != y_test[i]:
            errorCount += 1
    print('Error rate is',errorCount/y_test.shape[0])    
    return errorCount/y_test.shape[0]

def multiTest(n=10):
    errorSum = 0.0
    for k in range(n):
        errorRate = colicTest()
        errorSum += errorRate
    print('after %d iterations the average error rate is %f'%(n,errorSum/n))
    
if __name__ == '__main__':
    multiTest()