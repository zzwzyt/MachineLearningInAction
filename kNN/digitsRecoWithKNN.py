# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:51:08 2017

@author: User
"""

import numpy as np
import myKNN
import os

#将数据转换为向量
def imgtovec(filepath):
    dataList = []
    with open(filepath) as f:
        for currLine in f.readlines():
            currLine = list(currLine)
            currLine = [int(a) for a in currLine if a != '\n']
            dataList.extend(currLine)
        return np.array(dataList)
        
        
#加载文件夹中的数据
def loadDigit(filename):
    fileList = os.listdir(filename)
    m = len(fileList)
    dataArr = np.zeros((m,1024))
    label = np.zeros((m,))
    
    for i,subfile in enumerate(fileList):
        subfilename = filename+"\\"+subfile 
        sample = imgtovec(subfilename)
        dataArr[i] = sample
        label[i] = int(subfile[0])
    return dataArr,label

trainData,trainLabel = loadDigit('trainingDigits')
testData,testLabel = loadDigit('testDigits')

errorCount = 0
for i,inX in enumerate(testData):
    y_pred = myKNN.kNN(trainData,trainLabel,inX,k=5)
    if y_pred != testLabel[i]:
        print('Error predict:',y_pred,'The right answer:',testLabel[i])
        errorCount += 1
print('The error rate is ',errorCount/testLabel.shape[0])