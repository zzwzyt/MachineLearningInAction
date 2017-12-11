# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:35:56 2017

@author: User
"""

import numpy as np
import regTrees
import matplotlib.pyplot as plt

#模型树的叶节点生产函数: 节点保存的是权重
def linearSolve(dataSet):
    m,n = dataSet.shape
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    #X的第一个维度都设置为1，用于表示偏置项bias
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular,cannot do inverse, \n\
                        try increasing the second value of ops')
    ws = xTx.I*(X.T*Y)
    return ws,X,Y

#计算模型的权重
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

#计算模型误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return np.sum(np.power(Y-yHat,2))

if __name__ == '__main__':
    myMat = np.mat(regTrees.loadDataSet('exp2.txt'))
    myTree = regTrees.createTree(myMat,modelLeaf,modelErr,(1,10))
    x = myMat[:,0]
    y = myMat[:,1]
    spVal = myTree['spVal']
    index1 = np.nonzero(x<spVal)
    index2 = np.nonzero(x>=spVal)
    x1 = x[index1]
    x2 = x[index2]
    w2 = myTree['left']
    w1 = myTree['right']
    yHat1 = w1[1,0]*x1+w1[0,0]
    yHat2 = w2[1,0]*x2+w2[0,0]
    
    plt.figure()
    plt.scatter(x.A.flatten(),y.A.flatten())
    plt.plot(x1.T,yHat1.T,c='r')
    plt.plot(x2.T,yHat2.T,c='r')
    plt.show()


