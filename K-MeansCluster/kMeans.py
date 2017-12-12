# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:00:39 2017

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

def loadDataSet(filename):
    lineArr = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineList = line.strip().split('\t')
            lineList = [float(i) for i in lineList]
            lineArr.append(lineList)
    return lineArr

#计算欧氏距离
def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.square(vecA-vecB)))

#初始化聚类中心
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(np.max(dataSet[:,j]) - minJ)
        #np.random.rand(k,1)从[0,1]的均匀分布产生一个形状为（k,1)的随机数
        centroids[:,j] = minJ + rangeJ*np.random.rand(k,1)
    return centroids

def plotData(dataSet):
    plt.figure(1)
    plt.scatter(dataSet[:,0].A.flatten(),dataSet[:,1].A.flatten())
    plt.show()
    
#K均值聚类算法：
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    #自己实现
    m,n = dataSet.shape
    centroids = randCent(dataSet,k)
    dist = np.mat(np.zeros((m,k)))
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChanged = True
    while (clusterChanged):
        oldCentroids = centroids.copy()
        for i in range(m):
            for j in range(k):
                dist[i,j] = distEclud(dataSet[i,:],centroids[j,:])
            minDistIndex = np.argmin(dist[i,:])
            clusterAssment[i,:] = minDistIndex,np.min(dist[i,:])**2     
        for j in range(k):
            indexJ = np.nonzero(clusterAssment[:,0]==j)[0]
            clusterJ = dataSet[indexJ,:]
            centerC = np.mean(clusterJ,axis=0)
            centroids[j,:] = centerC
        if np.sum(oldCentroids-centroids) == 0:
            clusterChanged = False
    return centroids,clusterAssment

def kMeans1(dataSet,k,distMeas=distEclud,createCent=randCent):
    #教材实现
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
#        print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment

def biKMeans(dataSet,k,distMeas=distEclud):
    m,n = dataSet.shape
    clusterAssment = np.mat(np.zeros((m,2)))
    #创建初始簇
    centroid0 = np.mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(np.mat(centroid0),dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = np.inf
        for i in range(len(centList)):
            #尝试划分每一簇
            ptsIncurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss = kMeans1(ptsIncurrCluster,2,distMeas)
            #本次划分的误差
            
            sseSplit = np.sum(splitClustAss[:,1])
            #剩余数据的误差
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0],1])
#            print('sseSplit, and sseNotSplit:',sseSplit,sseNotSplit)
            #选择误差最小的划分
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
#        更新簇的分配结果
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
#        print('the bestCenToSplit is:',bestCentToSplit)
#        print('the len of bestClustAss is:',len(bestClustAss))
        #将被二分的簇的质心替换为二分完之后的两个质心
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
#        clusterAssment中被二分的簇的信息替换为二分之后簇的信息
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return centList,clusterAssment



dataMat = np.mat(loadDataSet('testSet.txt'))
#plotData(dataMat)
#centroids = randCent(dataMat,4)
myCentroids,clusterAssing = kMeans(dataMat,4)

y = clusterAssing[:,0].A.flatten()
plt.figure(2)
plt.scatter(dataMat[:,0].A.flatten(),dataMat[:,1].A.flatten(),c=y)

myCentroids1,clusterAssing1 = kMeans1(dataMat,4)
y1 = clusterAssing1[:,0].A.flatten()
plt.figure(3)
plt.scatter(dataMat[:,0].A.flatten(),dataMat[:,1].A.flatten(),c=y1)
plt.show()

dataMat3 = np.mat(loadDataSet('testSet2.txt'))
centList,myNewAssments = biKMeans(dataMat3,3)
y3 = myNewAssments[:,0].A.flatten().tolist()
centList = np.mat([matri.A.flatten().tolist() for matri in centList])
plt.figure(4)
plt.scatter(dataMat3[:,0].A.flatten(),dataMat3[:,1].A.flatten(),c=y3)
plt.scatter(centList[:,0].A.flatten(),centList[:,1].A.flatten(),marker='+',s=200,c='r')
plt.show()