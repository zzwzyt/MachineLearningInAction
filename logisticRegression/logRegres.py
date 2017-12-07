# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:24:04 2017

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
 
def loadDataSet():
    dataMat = []
    labelMat = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            #第一个维度设置为1，相当于bias
            dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return np.array(dataMat),labelMat

def sigmoid(inX):
    return 1/(1+np.exp(-inX))

#梯度上升优化算法
#自己实现
def gradAscent(dataMat,labelMat,alpha=0.001,iterNum=500):
    """其中dataMat是二维数组，labelMat是shape为（m,）的数组或长度为m的列表，
    alpha是学习率，iterNum是迭代次数。代价函数对weights求导的方向导数是
    np.dot(X.T,（y_hat-labelMat))"""
    m,n = dataMat.shape
    weights = np.ones(n)
    for i in range(iterNum):
        y_hat = sigmoid(np.dot(dataMat,weights))
        h = labelMat - y_hat
        delta = np.dot(dataMat.T,h)
        weights += alpha*delta
    return weights
#教材实现
#def gradAscent(dataMatIn, classLabels):
#    dataMatrix = np.mat(dataMatIn)             #convert to NumPy matrix
#    labelMat = np.mat(classLabels).transpose() #convert to NumPy matrix
#    m,n = np.shape(dataMatrix)
#    alpha = 0.001
#    maxCycles = 500
#    weights = np.ones((n,1))
#    for k in range(maxCycles):              #heavy on matrix operations
#        h = sigmoid(dataMatrix*weights)     #matrix mult
#        error = (labelMat - h)              #vector subtraction
#        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
#    return weights

#画出决策边界：
def plotBestFit(dataMat,labelMat,weights):
   
#    y_hat = sigmoid(np.dot(dataMat,weights))
    x1 = np.arange(-3.0,3.0,0.1)
    #0是两个分类（类别1和0）的分界处，因为y为0时，sigmoid(y)=0.5,小于0.5分类为0，大于0.5分类为1.
    #因此，画决策边界时，可设定0=w0x0+w1x1+w2x2，其中w0是bias
    x2 = (-weights[0]-weights[1]*x1)/weights[2]
    plt.figure(1)
    plt.scatter(dataMat[:,1],dataMat[:,2],c=labelMat)
    plt.plot(x1,x2,c='r')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#训练算法：随机梯度上升
def stocGradAscent(dataMat,labelMat,alpha=0.01):
    m,n = dataMat.shape
    weights = np.ones(n)
    costs = []
    for i in range(m):
        h = sigmoid(np.sum(dataMat[i,:]*weights))
        error = labelMat[i] - h
        weights += alpha*dataMat[i]*error
        y_hat = sigmoid(np.dot(dataMat,weights))
        cost = np.sum(np.square(y_hat-labelMat))
        costs.append(cost)
    return weights,costs

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    costs = []
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #alpha decreases with iteration, does not go to 0 because of the constant
            #alpha随着迭代次数不断减小，会缓解函数的波动情况
            alpha = 4/(1.0+j+i)+0.0001    
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
        y_hat = sigmoid(np.dot(dataMatrix,weights))
        cost = np.sum(np.square(y_hat-classLabels))
        costs.append(cost)
    return weights,costs

def predict(inX,weights):
    y_pred = np.zeros(inX.shape[0])
    y_reg = sigmoid(np.dot(inX,weights))
    index = y_reg>0.5
    y_pred[index] = 1.0
    return y_pred
    
if __name__ == '__main__':
    dataMat,labelMat = loadDataSet()
    weights,costs = stocGradAscent(dataMat,labelMat)
    plotBestFit(dataMat,labelMat,weights)
    weights1,costs1 = stocGradAscent1(dataMat,labelMat)
    plotBestFit(dataMat,labelMat,weights1)
    
    #画出代价函数
    plt.figure(2)
    plt.plot(costs)
    plt.show
    
    plt.figure(3)
    plt.plot(costs1)
    plt.show