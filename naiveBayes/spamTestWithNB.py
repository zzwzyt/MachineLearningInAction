# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:06:11 2017

@author: User
"""

import numpy as np
import myBayes
from sklearn.model_selection import train_test_split

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        with open('email/spam/%d.txt'%i,errors='ignore') as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
        with open('email/ham/%d.txt'%i,errors='ignore') as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
    vocabList = myBayes.createVocabList(docList)
    numWords = len(vocabList)
    dataArr = np.zeros((50,numWords))
    for i,item in enumerate(docList):
        returnVec = myBayes.bagOfWords2VecMN(vocabList,item)
        dataArr[i] = returnVec
    
    errorCount = 0.0
    for k in range(10):
        
        X_train,X_test,y_train,y_test = train_test_split(dataArr,classList,train_size=0.8)
        p0V,p1V,pSpam = myBayes.trainNB1(X_train,y_train)
        y_pred = []
        for i,item in enumerate(X_test):
            y_hat = myBayes.classifyNB(item,p0V,p1V,pSpam)
            y_pred.append(y_hat)
            if y_hat != y_test[i]:
                errorCount += 1
        print(errorCount/((k+1)*len(y_test)))

if __name__ == '__main__':
    spamTest()
        