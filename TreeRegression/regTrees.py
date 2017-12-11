# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:10:51 2017

@author: User
"""

import numpy as np

#CART算法的实现代码
def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = [float(i) for i in curLine]
            dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])     #负责生成叶节点

def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0] #总方差=均方差*样本个数

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]    #tolS是容许的误差下降值
    tolN = ops[1]    #tolN是切分的最少样本数
    #如果所有的值相等，则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = dataSet.shape
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].A.flatten().tolist()):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            #如果某个子集的大小小于参数tolN，那么就不应切分
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

#回归树后剪枝函数
def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree,testData):
    if testData.shape[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    #如果左右分支都不再是子树，则考虑合并
    if not isTree(tree['right']) and not isTree(tree['left']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) + \
                       np.sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print('Merging')
            return treeMean
        else:
            return tree
    else:
        return tree
        
        
if __name__ == '__main__':
    #testMat = np.mat(np.eye(4))
    #mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    #myDat = loadDataSet('ex00.txt')
    #myMat = np.mat(myDat)
    #myTree = createTree(myMat)
    myDat2 = loadDataSet('ex2.txt')
    myTree2 = createTree(np.mat(myDat2),ops=(0,1))
    myDat2Test = np.mat(loadDataSet('ex2test.txt'))
    myTree22 = prune(myTree2,np.mat(myDat2Test))
