# -*- coding: utf-8 -*-

from numpy import *
 
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)
 
def pca(dataMat, topNfeat=9999999):    #topNfeat 降维后的维度
    meanVals = mean(dataMat, axis=0)   #按列求均值，即每一列求一个均值，不同的列代表不同的特征
    #print meanVals                
    meanRemoved = dataMat - meanVals   #去均值，将样本数据的中心点移到坐标原点
    print ("将样本数据的中心点移到坐标原点",meanRemoved);
    covMat = cov(meanRemoved, rowvar=0)         #计算协方差矩阵
    #print covMat                             
    eigVals,eigVects = linalg.eig(mat(covMat))  #计算协方差矩阵的特征值和特征向量
    #print eigVals
    #print eigVects
    eigValInd = argsort(eigVals)                #sort, sort goes smallest to largest  #排序，将特征值按从小到大排列
    #print eigValInd
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions      #选择维度为topNfeat的特征值
    #print eigValInd
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest   #选择与特征值对应的特征向量
    print ("选择与特征值对应的特征向量",redEigVects)
    lowDDataMat = meanRemoved * redEigVects   #transform data into new dimensions    #将数据映射到新的维度上，lowDDataMat为降维后的数据
    print ("降维后的数据",lowDDataMat)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals         #对原始数据重构，用于测试
    print ("对原始数据重构，用于测试",reconMat)
    return lowDDataMat, reconMat
 
def replaceNanWithMean():             #均值代替那些样本中的缺失值
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number) # .A表示把矩阵转化为数组array
        #nonzero(~isnan(datMat[:,i].A))[0] 返回非0元素所在行的索引； 
        #>>> nonzero([True,False,True])
        #    (array([0, 2]),) 第0个和第3个元素非0
        #~isnan()返回Ture or False
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
#：https://blog.csdn.net/qq_29422251/article/details/79279446


result=pca([[1,1],[2,2]],topNfeat=2)
print("降维后的数据,重构后的数据",result)
