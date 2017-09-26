'''
Created on 2017年9月22日

@author: chc
'''

from numpy import *
from os import listdir
from os.path import isfile, join

maxSeqLength = 22
numFiles = 926
def pca(dataMat, topNfeat):
    meanVals = mean(dataMat, axis=0)
    DataAdjust = dataMat - meanVals           #减去平均值
    covMat = cov(DataAdjust, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat)) #计算特征值和特征向量
    #print eigVals
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]   #保留最大的前K个特征值
    redEigVects = eigVects[:,eigValInd]        #对应的特征向量
    lowDDataMat = DataAdjust * redEigVects     #将数据转换到低维新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals   #重构数据，用于调试
    return lowDDataMat, reconMat

semanticVectors = loadtxt('semanticVector.txt')
semanticVectors = mat(semanticVectors, dtype = float32)
print(semanticVectors.shape)

syntaxVectors = loadtxt('syntaxVector.txt')
syntaxVectors = mat(syntaxVectors, dtype = float32)
print(syntaxVectors.shape)

statementVec = zeros([semanticVectors.shape[0], 10000])
for i in range(semanticVectors.shape[0]):   
    s = semanticVectors[i,:].T
    #print(s.shape)
    y = syntaxVectors[i,:]
    #print(y.shape)
    mul = s * y
    mul = reshape(mul, (1, -1))
    #print(mul.shape)
    statementVec[i] = mul
    #print(mul)
    
s, c = pca(statementVec, 250)
statementVectors = array(s, dtype = float32)
save('statementVectors', statementVectors)
print(statementVectors.shape)

count = 0
statementList = ['x' for n in range(1102)]
for line in open("statementList.txt"): 
    statementList[count] = line.rstrip()
    count = count + 1

positiveFiles = ['positiveFiles/' + f for f in listdir('positiveFiles/') if isfile(join('positiveFiles/', f))]
negativeFiles = ['negativeFiles/' + f for f in listdir('negativeFiles/') if isfile(join('negativeFiles/', f))]
ids = zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
for pf in positiveFiles:
    indexCounter = 0
    for line in open(pf):
        word = line.rstrip()
    
        try:
            ids[fileCounter][indexCounter] = statementList.index(word)
        except ValueError:
            ids[fileCounter][indexCounter] = 399999  # Vector for unkown words
        indexCounter = indexCounter + 1
        #if indexCounter >= maxSeqLength:
        #   break
    fileCounter = fileCounter + 1 
 
for nf in negativeFiles:
    indexCounter = 0
    for line in open(nf):
        
        word = line.rstrip()
    
        try:
            ids[fileCounter][indexCounter] = statementList.index(word)
        except ValueError:
            ids[fileCounter][indexCounter] = 399999  # Vector for unkown words
        indexCounter = indexCounter + 1
        #if indexCounter >= maxSeqLength:
        #   break
    fileCounter = fileCounter + 1 
print(ids[1])
#save('idsMatrix', ids)






