# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 07:41:30 2017

@author: User
"""

import numpy as np
from sklearn.preprocessing import scale

#标准回归函数和数据导入函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    #返回列表dataMat和labelMat
    return dataMat,labelMat

def standRegre(xArr,yArr):
    #列表转换为矩阵：如果列表是二维的，则会将列表中的每个元素当成矩阵的一行；如果列表是一维的，则会将列表本身当作矩阵的第一行
    #xArr是长度为200的列表，每个元素都是一个二维列表，转换为矩阵后的形状就是200*2
    xMat = np.mat(xArr)
    #yArr是长度为200的列表，每个元素都是一个数字，转换为矩阵后的形状就是1*200，转置后形状为200*1
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    #推导w = (X.T*X).I*X.T*y
    #如果X.T*X不存在逆矩阵则会出错
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

#局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k = 1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2*k**2))[0,0]
    xTx = xMat.T * (weights*xMat)
    if np.linalg.det(xTx) == 0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k = 1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#计算预测误差
def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()

#岭回归
#计算回归系数
def ridgeRegress(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    #岭回归得到的回归系数w = (X.T*X + lam*E).I*X.T*y
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

#在一组lam上测试结果
def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    #数据标准化
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat,0)
    xVar = np.var(xMat,0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat,yMat,np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat

#前向逐步线性回归
#eps是每次迭代需要调整的步长，numIt表示迭代次数
def stageWise(xArr,yArr,eps = 0.01,numIt = 100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean  #y的方差不需要设置为1
    #xMat的标准化,regularize是自定义的，两个计算有什么区别？
#    xMat = regularize(xMat)
    xMat = scale(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    #将所有权重初始化为1
    ws = np.zeros((n,1))
    #为了实现贪心算法建立ws的两份副本
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
#        print(ws.T)
        lowestError = np.inf
        #遍历所有特征
        for j in range(n):
            #分别计算增加或减少该特征对误差的影响
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = np.mat(xMat) * np.mat(wsTest)
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat
