import numpy as np

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]),float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return np.array(dataMat),np.array(labelMat)

def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>=H:
        aj = H
    elif aj<=L:
        aj = L
    return aj

#简化版SMO算法
def smoSimple(dataArr,labelArr,C,toler,maxIter):
    
    m,n = dataArr.shape
    alphas = np.zeros(m)    #初始化所有alpha为0
    b = 0
    iterNum = 0
    
    while(iterNum<maxIter):
        alphaPairsChanged = 0
        #遍历整个数据集
        for i in range(m):
#           ui=wxi+b,其中w = w = a1*y1*x1+a2*y2*x2+...+am*ym*xm
            w = np.dot(alphas*labelArr,dataArr)
            ui = float(np.dot(w,dataArr[i,:]) + b)
            Ei = ui - float(labelArr[i])    #预测的类别和真实类别之差，即误差
            #toler是容错率，如果误差很大，则对该数据实例所对应的alpha值进行优化??
            if ((labelArr[i]*Ei < -toler) and (alphas[i] < C)) or ((labelArr[i]*Ei > toler) and\
                (alphas[i] > 0)):
                j = selectJrand(i,m)
                alphaIold = alphas[i].copy()    #浅复制，为alphaIold分配新的内存
                alphaJold = alphas[j].copy()
                uj = float(np.dot(w,dataArr[j].T) + b)
                Ej = uj - float(labelArr[j])
                #保证alpha在0与C之间
                if (labelArr[i] != labelArr[j]):        #如果实例i和j的标签不一样，即在超平面的两侧
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:                                   #实例i和j的标签一样，即在超平面的同一侧
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                    print("L==H")
                    continue
                #eta是对alphaJ的二阶导数
                eta = -np.dot(dataArr[i],dataArr[i]) - np.dot(dataArr[j],dataArr[j]) + 2*np.dot(dataArr[i],dataArr[j])
                if eta >= 0:
                    print('eta >= 0')
                    continue
                
                alphas[j] = alphas[j] - labelArr[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
#                    print('j not moving enough')
                    continue
                alphas[i] = alphas[i] + labelArr[i]*labelArr[j]*(alphaJold - alphas[j])
                
                b1 = b - Ei - labelArr[i]*(alphas[i]-alphaIold)*np.dot(dataArr[i].T,dataArr[i]) - \
                     labelArr[j]*(alphas[j]-alphaJold)*np.dot(dataArr[i].T,dataArr[j])
                b2 = b - Ej - labelArr[i]*(alphas[i]-alphaIold)*np.dot(dataArr[i].T,dataArr[j]) - \
                     labelArr[j]*(alphas[j]-alphaJold)*np.dot(dataArr[j].T,dataArr[j])
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else:   b = (b1+b2)/2.0
                
                alphaPairsChanged += 1
                print('iter: %d i:%d, pairs changed %d' %(iterNum,i,alphaPairsChanged))
        if (alphaPairsChanged == 0):
             #变量iterNum存储的是在没有任何alpha改变的情况下遍历数据集的次数
            #当该变量达到输入值maxIter时，函数结束运行并退出
            iterNum += 1
        else:
            iterNum == 0
        print('iteration number: %d'%iterNum)
    return b,alphas

def kernelTrans(X,A,kTup):
    m,n = X.shape
    K = np.zeros((m,1))
    if kTup[0] == 'lin':
        K = np.dot(X,A.T)
    if kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = np.dot(deltaRow,deltaRow.T)
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We have a problem...that kernel is not recoginized')
    return K

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler,kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        #eCache的第一列给出的是eCache是否有效的标志位，而第二列给出的是实际的E值
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and \
        (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - \
             oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - \
             oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iterNum = 0
    entireSet = True; alphaPairsChanged = 0
    while (iterNum < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iterNum,i,alphaPairsChanged))
            iterNum += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iterNum,i,alphaPairsChanged))
            iterNum += 1
        if entireSet:  #toggle entire set loop
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iterNum)
    return oS.b,oS.alphas

#利用核函数进行分类的径向基测试函数
def testRbf(k1 = 1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')   #与上面的数据集不同
    errorCount = 0
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        #决策函数f(x)=sign(∑aiyik(x,xi)+b),其中a和b都是最优化后得到的
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

def calcWs(alphas,dataArr,labelArr):
    m,n = dataArr.shape
    weights = np.zeros((n,1))
    weights = np.dot(alphas*labelArr,dataArr)
    return weights
    
if __name__ == '__main__':
    testRbf()
    # dataArr,labelArr = loadDataSet('testSet.txt')
    # b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
    # weights = calcWs(alphas,dataArr,labelArr)
    # x1 = range(-2,10)
    # x2 = (-b - weights[0]*x1)/weights[1]
    #
    # sv = []
    # for i in range(len(alphas)):
    #     if alphas[i] > 0:
    #       sv.append(dataArr[i])
    # sv = np.array(sv)
    #
    # import matplotlib.pyplot as plt
    # plt.scatter(sv[:,0],sv[:,1],c='r',marker='o',s=100)
    # plt.scatter(dataArr[:,0],dataArr[:,1],c=labelArr)
    # plt.plot(x1,x2)
    #
    # plt.show()
    
