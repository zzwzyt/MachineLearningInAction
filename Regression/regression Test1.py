# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 07:59:57 2017

@author: User
"""

import regression
import numpy as np
import matplotlib.pyplot as plt

xArr,yArr = regression.loadDataSet('ex0.txt')

#ws = regression.standRegre(xArr,yArr)
#
#xMat = np.mat(xArr)
#yMat = np.mat(yArr)
#yHat = xMat * ws
#
#print(np.corrcoef(yHat.T,yMat))
#
##画出散点图和拟合直线
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
#
#xCopy = xMat.copy()
#xCopy.sort(0)
#yHat = xCopy*ws
#ax.plot(xCopy[:,1],yHat,c='red')
#plt.show()
##画出散点图和拟合直线2
#plt.figure()
#plt.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
#xCopy = xMat.copy()
#xCopy.sort(0)
#yHat = xCopy*ws
#plt.plot(xCopy[:,1],yHat,c='red')
#plt.show()

#print(regression.lwlr(xArr[0],xArr,yArr,1.0))
#print(regression.lwlr(xArr[0],xArr,yArr,0.001))
#print(regression.lwlrTest(xArr,xArr,yArr,0.03))
#print(np.corrcoef(regression.lwlr(xArr[0],xArr,yArr,1.0),yArr[0]))
#print(np.corrcoef(regression.lwlr(xArr[0],xArr,yArr,0.001),yArr[0]))
yHat = regression.lwlrTest(xArr,xArr,yArr,0.01)
xMat = np.mat(xArr)
srtInd = xMat[:,1].argsort(0)
#srtInd是二维matrix,故xMat变成3维
#等价于以下语句
xSort = xMat[np.array(srtInd).reshape((1,-1))[0]]
xSort = xMat[srtInd][:,0,:]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=2,c='red')
plt.show()

#a = np.arange(8)
#b = a.reshape((4,2))
#srtind = b[:,1].argsort(0)
#c = b[srtind]
#
#b = np.mat(b)
#srtind = b[:,1].argsort(0)
#c = b[srtind]
#
abX,abY = regression.loadDataSet('abalone.txt')
#print(regression.stageWise(xArr,yArr,0.01,200))
ridgeweights = regression.ridgeTest(abX,abY)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeweights)
plt.show()

returnMat = regression.stageWise(abX,abY,0.01,1000)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(returnMat)
plt.show()





























