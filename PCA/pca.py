# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:59:38 2017

@author: User
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName,delim='\t'):
    datArr = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split(delim)
            curLine = [float(i) for i in curLine]
            datArr.append(curLine)    
    return np.mat(datArr)

def pca(dataMat,topNfeat=9999999):
    meanVals = np.mean(dataMat,axis=0)
    X = dataMat - meanVals
    #rowvar=1/True（默认）则每行代表一个特征
    #rowvar=0时则每列代表一个特征
    covMat = np.cov(X,rowvar=0)
    eigVal, eigVec = np.linalg.eig(covMat)
    #特征值从小到大排序取前N个
    eigIndex = np.argsort(eigVal)
    eigIndexCropped = eigIndex[:-(topNfeat+1):-1] 
    redEigVec = eigVec[:,eigIndexCropped]
    #用N个特征将原始数据转换到新空间中
    #lowDDataMat是X在低维空间redEigVec中的投影
#    print(X.shape,redEigVec.shape)
    lowDDataMat = X*redEigVec
    #原始数据被重构后返回用于调试
    reconMat = (lowDDataMat*redEigVec.T) + meanVals
    
    return lowDDataMat,reconMat

dataMat = loadDataSet('testSet.txt')
loDMat,reconMat = pca(dataMat,1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].A.flatten(),dataMat[:,1].A.flatten(),marker='^',s=90)
ax.scatter(reconMat[:,0].A.flatten(),reconMat[:,1].A.flatten(),\
           marker='o',s=50,c='r')


def replaceNanWithMean():
    dataMat = loadDataSet('secom.data',' ')
    numFeat = dataMat.shape[1]
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:,i].A))[0],i])
        dataMat[np.nonzero(np.isnan(dataMat[:,i].A))[0],i] = meanVal
    return dataMat

dataMat = replaceNanWithMean()
meanVals = np.mean(dataMat,axis=0)
meanRemoved = dataMat - meanVals
covMat = np.cov(meanRemoved,rowvar=0)
eigVals,eigVects = np.linalg.eig(np.mat(covMat))
eigVals = eigVals[:20,]
eigValsSum = np.sum(eigVals)
plt.figure()
plt.scatter(list(range(eigVals.shape[0])),eigVals/eigValsSum,marker='^')
plt.plot(list(range(eigVals.shape[0])),eigVals/eigValsSum,c='r')
plt.show()















