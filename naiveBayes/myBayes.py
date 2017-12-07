# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:28:07 2017

@author: User
"""

import numpy as np

def loadDataSet():
    sentenses = ['my dog has flea problems help please','maybe not take him to dog park stupid',
                 'my dalmation is so cute I love him','stop posting stupid worthless garbage',
                 'mr licks ate my steak how to stop him','quit buying worthless dog food stupid']
    classVec = [0,1,0,1,0,1]
    senList = []
    for sent in sentenses:
        sl = sent.split(' ')
        senList.append(sl)
    return senList,classVec
    
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #其中|表示取并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
#    #自己实现
#    m = len(vocabList)
#    iniVec = np.zeros((m,))
#    for i,item in enumerate(vocabList):
#        if item in inputSet:
#            iniVec[i] = 1
#    return iniVec
    #教材实现
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('The word: %s is not in my Vocabulary!'%word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs,numWords = trainMatrix.shape
    pAbusive = sum(trainCategory)/float(numTrainDocs) #类别为1的样本的占比，只有0，1二分类才可以这么求
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            #统计分类为1时，各个词出现的次数
            p1Num += trainMatrix[i]
            #将所有被分类为1的文档中的词出现的次数加和
#            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
#            p0Denom += sum(trainMatrix[i])
    p1Denom = np.sum(p1Num)
    p0Denom = np.sum(p0Num)
    #计算条件概率：分类为1/0时，各个词出现的概率
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

#为避免概率为0，可以将所有次的出现次数初始化为1，并将分母初始化为2
def trainNB1(trainMatrix,trainCategory):
    numTrainDocs,numWords = trainMatrix.shape
    pAbusive = sum(trainCategory)/float(numTrainDocs) #类别为1的样本的占比，只有0，1二分类才可以这么求
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            #统计分类为1时，各个词出现的次数
            p1Num += trainMatrix[i]
            #将所有被分类为1的文档中的词出现的次数加和
#            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
#            p0Denom += sum(trainMatrix[i])
    p1Denom = np.sum(p1Num)
    p0Denom = np.sum(p0Num)
    #计算条件概率：分类为1/0时，各个词出现的概率
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive 

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #此处p1Vec = log(p1Num/p1Denom)
    p1 = np.sum(vec2Classify*p1Vec) + np.log(pClass1) #np.sum(np.log(pClass1*Vec2Classify*p1Num/p1Denom))
    p0 = np.sum(vec2Classify*p0Vec) + np.log(1.0 - pClass1) #即先取对数再求和
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList,inputSet):
#        #自己实现
#    m = len(vocabList)
#    iniVec = np.zeros((m,))
#    for i,item in enumerate(vocabList):
#        if item in inputSet:
#            iniVec[i] = inputSet.count(item)
#    return iniVec
#    教材实现
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('The word: %s is not in my Vocabulary!'%word)
    return returnVec

if __name__ == '__main__':
    listOfPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    trainMat = np.array(trainMat)
    p0V,p1V,pAb = trainNB1(trainMat,listClasses)
    testEntry = ['love','my','dalmation']
    testVec = setOfWords2Vec(myVocabList, testEntry)
    print(classifyNB(testVec,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    testVec = setOfWords2Vec(myVocabList, testEntry)
    print(classifyNB(testVec,p0V,p1V,pAb))
    testEntry = ['love','my','dalmation','love']
    testVec = setOfWords2Vec(myVocabList, testEntry)
    testVec1 = bagOfWords2VecMN(myVocabList, testEntry)