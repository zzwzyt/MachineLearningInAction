# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:24:30 2017

@author: User
"""

import numpy as np

def loadSimpData():
    datMat = np.mat([[1.,2.1],[2.,1.1],[1.3,1.],
                     [1.,1.],[2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

#单层决策树生成函数
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        #所有在阀值一边的数据会分到类别-1，另外一般的数据分到类别+1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

##生成具有最低错误率的单层决策树
def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    #minError初始化为正无穷大
    minError = np.inf   #init error sum, to +infinity
    #遍历每个特征
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max();
        #根据最小值和最大值来确定步长
        stepSize = (rangeMax-rangeMin)/numSteps
#        print(len(stepSize))
        #遍历每个步长,阀值可从rangeMin取值到rangeMax
        #i=0时，threshVal=rangeMin,i=numSteps时，threshVal=rangeMax
        for j in range(-1,int(numSteps)+1):     #loop over all range in current dimension
            #在大于和小于之间切换不等式    
            for inequal in ['lt', 'gt']:        #go over less than and greater than
                    #设置阀值thresh value
                    threshVal = (rangeMin + float(j) * stepSize)
                    predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                    #构建列向量，如果predictedVals中的值不等于labelMat中真正类别的值，那么
                    #errArr对应位置的值为1，否则为0
                    errArr = np.mat(np.ones((m,1)))
                    errArr[predictedVals == labelMat] = 0
#                    print("predictedVals",predictedVals.T,"errArr",errArr.T)
                    #weihtedError = 权重向量D*错误向量errArr
                    weightedError = D.T*errArr  #calc total error multiplied by D
#                    print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"\
#                          % (i, threshVal, inequal, weightedError))
                    if weightedError < minError:
                        minError = weightedError
                        #此处要使用copy(),否则后面predictedVals的变更会直接影响bestClasEst的值
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

#基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):      #用户可指定迭代次数
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)                        #init D to all equal
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)        #build Stump
        #print "error",error
        #α= 1/2*ln((1-ε)/ ε)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))     #calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
#        print("alpha",alpha)
        weakClassArr.append(bestStump)#store Stump Params in Array
#        print("classEst",classEst)
        #以下三行是D的迭代，D（t+1）= D(t)*exp**expon/sum(D)
        #样本正确分类expon=-alpha，错误分类expon=alpha
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = np.multiply(D,np.exp(expon)) #Calc New D, element-wise 
        D = D/D.sum()
#        print("D",D)
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        #根据每个分类器的分类结果和对应的权重来集成计算最终的分类结果
        aggClassEst += alpha*classEst
#        print("aggClassEst",aggClassEst)
        #aggClassEst和classLabels不一致的设置为1，一致的设置为0
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        #print aggErrors
        errorRate = aggErrors.sum()/m
        print('totla error: ',errorRate,'\n')
        #如果总错误率达到0.0，则提前结束循环
        if errorRate == 0.0: 
            break
    return weakClassArr,aggClassEst

#AdaBoost分类函数
def adaClassify(datToClass,classifierArr):
    dataMat = np.mat(datToClass)
    m = dataMat.shape[0]
    aggClassEst = np.zeros((m,1))
    for i,classifier in enumerate(classifierArr):
        classEst = stumpClassify(dataMat,classifier['dim'],classifier['thresh'],classifier['ineq'])
#        print(classEst,classifier['alpha'])
        aggClassEst += classifier['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)      

#ROC曲线的绘制及AUC计算函数
#predStrengths代表的是分类器的预测强度
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    #cur保留绘制光标的位置，从右上角开始往左下角画
    cur = (1.0,1.0) #cursor
    #ySum用于计算AUC的值
    ySum = 0.0 #variable to calculate AUC
    #过滤数组，计算正例的数目
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    #获取排序索引，按照最小到最大的顺序排列的
    sortedIndicies = predStrengths.argsort() #get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    #构建画笔ax
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        #每得到一个标签为1.0的类，则要沿着y轴的方向下降一个步长，即不断降低真阳率？？
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep;
        #对于每个其它类别的标签，则是在x轴方向上倒退一个步长
        else:
            delX = xStep
            delY = 0;
            #求高度和
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    #将高度和ySum乘以x轴的步长xStep即可得到AUC
    print("the Area Under the Curve is: ",ySum*xStep)

#自适应数据加载函数
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
   
if __name__ == '__main__':  
#    datMat,classLabels = loadSimpData()
#    classifierArr,aggClassEst = adaBoostTrainDS(datMat,classLabels,30)
#    print(adaClassify([[5,5],[0,0]],classifierArr))
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray,aggClassEst = adaBoostTrainDS(datArr,labelArr,10)
    plotROC(aggClassEst.T,labelArr)
    
    