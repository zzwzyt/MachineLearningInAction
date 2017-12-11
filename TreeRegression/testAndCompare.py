# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:36:05 2017

@author: User
"""

import numpy as np
import regTrees
import modelTrees

#用树回归进行预测的代码
def regTreeEval(model,inDat):
    return model

def modelTreeEval(model,inDat):
    n = inDat.shape[1]
    #X的第一维代表偏置项bias
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not regTrees.isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if regTrees.isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if regTrees.isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData) 

def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat

if __name__ == '__main__':
    trainMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = regTrees.createTree(trainMat,ops=(1,20))
    yHat = createForeCast(myTree,testMat[:,0])
    print('regTrees',np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])
    myTree2 = regTrees.createTree(trainMat,modelTrees.modelLeaf,modelTrees.modelErr,ops=(1,20))
    yHat2 = createForeCast(myTree2,testMat[:,0],modelTreeEval)
    print('modelTrees',np.corrcoef(yHat2,testMat[:,1],rowvar=0)[0,1])
    
    indexSored = np.argsort(testMat[:,0].A.flatten()).tolist()
    x = testMat[:,0].A.flatten()
    y = testMat[:,1].A.flatten()
    xSorted = x[indexSored]
    ySorted = yHat2[indexSored]
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(testMat[:,0].A.flatten(),testMat[:,1].A.flatten())
    plt.plot(xSorted,ySorted,c='r')
    plt.show()

