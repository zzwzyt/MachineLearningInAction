# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:18:28 2017

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:43:57 2017

@author: User
"""

import numpy as np

#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
#    自己实现
    labels = [example[-1] for example in dataSet]
    m = len(labels)
    labelCounts = {}
    for i,currentLabel in enumerate(labels):
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0) + 1 
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key])/m
        shannonEnt -= prob*np.log2(prob)
    return shannonEnt

#    #教材实现
#    numEntries = len(dataSet)
#    labelCounts = {}
#    for featVec in dataSet:
#        currentLabel = featVec[-1]
#        if currentLabel not in labelCounts.keys():
#            labelCounts[currentLabel] = 0
#        labelCounts[currentLabel] += 1
#    shannonEnt = 0.0
#    for key in labelCounts.keys():
#        prob = float(labelCounts[key])/numEntries
#        shannonEnt -= prob*np.log2(prob)      
#    return shannonEnt
    
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    featureNames = ['no surfacing','flippers']
    return dataSet,featureNames

#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    """"按信息增益率选择划分特征，属于C4.5"""
    #自己实现
    bestFeature = -1
    bestInfoGainRate = 0.0
    shannonEntD = calcShannonEnt(dataSet)
    for axis in range(len(dataSet[0])-1):
        newEntropy = 0
        #先转换成np.array会有问题，因为array中只能包含同一类型的数据，float会被转换为string
        #这样在splitDataSet()中判断条件if featVec[axis] == value会出问题
#        dataList = np.array(dataSet)[:,axis].flatten().tolist()
        dataList = [example[axis] for example in dataSet]
        for value in set(dataList):
            valueProb = dataList.count(value) / float(len(dataSet))
            retDataSet = splitDataSet(dataSet,axis,value)
            newEntropy += valueProb*calcShannonEnt(retDataSet)  
        gainRate = (shannonEntD - newEntropy)/shannonEntD
        if gainRate > bestInfoGainRate:
            bestInfoGainRate = gainRate
            bestFeature = axis
    return bestFeature

#    #教材实现
#    numFeatures = len(dataSet[0]) - 1
#    baseEntropy = calcShannonEnt(dataSet)
#    bestInfoGain = 0.0
#    bestFeature = -1
#    for i in range(numFeatures):
#        featList = [example[i] for example in dataSet]
#        uniqueVals = set(featList)
#        newEntropy = 0.0
#        for value in uniqueVals:
#            #按不同的值对数据集特征进行划分
#            subDataSet = splitDataSet(dataSet,i,value)
#            prob = len(subDataSet)/float(len(dataSet))
#            newEntropy += prob*calcShannonEnt(subDataSet)
#        #信息增益 = 整个数据集的熵 - 以第i个特征划分数据集得到的熵
#        infoGain = baseEntropy - newEntropy
#        #信息增益最大特征的为最优
#        if infoGain > bestInfoGain:
#            bestInfoGain = infoGain
#            bestFeature = i
#    return bestInfoGain,bestFeature

#如果数据集已经处理了所有属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该
#叶子节点次数，通常可采用多数表决的方法决定该叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for value in set(classList):
        counts = classList.count(value)
        classCount[value] = counts
    sortedClassCount = sorted(classCount.items(),key=lambda x: (x[1], x[0]),reverse=True)    
    return sortedClassCount[0][0]   

#创建树
def createTree(dataSet,featureNames):
    classList = [example[-1] for example in dataSet]
    #如果类别完全相同，则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有的特征时，返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatName = featureNames[bestFeat]
    myTree = {bestFeatName:{}}
    del (featureNames[bestFeat])
    #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subFeatureNames = featureNames[:]
        myTree[bestFeatName][value] = createTree(splitDataSet(dataSet,bestFeat,value),subFeatureNames)
    return myTree

#使用决策树执行分类
def classify(inputTree,featureNames,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    #获取标签字符串的索引
    featIndex = featureNames.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featureNames,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    #pickle（除了最早的版本外）是二进制格式的，所以你应该带 'b' 标志打开文件
#    fw = open(filename,'w') 
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    with open(filename,'rb') as fr:
        return pickle.load(fr)

if __name__ == '__main__':
    dataSet,featureNames = createDataSet()
    featNames = featureNames.copy()
    #print(calcShannonEnt(dataSet))
    #print(splitDataSet(dataSet,0,1))
    #print(chooseBestFeatureToSplit(dataSet))
    myTree = createTree(dataSet,featureNames)
    print(classify(myTree,featNames,[1,0]))
    storeTree(myTree,'myTreeStorage.txt')
    print(grabTree('myTreeStorage.txt'))