from numpy import *
def pca(dataMat, topNfeat=9999999):    #topNfeat 降维后的维度
    #去均值，将样本数据的中心点移到坐标原点
    meanVals = mean(dataMat, axis=0)#按列求均值，即每一列求一个均值，不同的列代表不同的特征 
    meanRemoved = dataMat - meanVals  
    print ("将样本数据的中心点移到坐标原点\n",meanRemoved);
    
    #计算协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)         
    print ("协方差矩阵:\n",covMat)                

    #计算协方差矩阵的特征值和特征向量    
    #确保方差最大  ，构造新特征两两独立       
    eigVals,eigVects = linalg.eig(mat(covMat)) 
    print ("特征值\n",eigVals)
    print ("特征向量\n",eigVects)#特征向量构成了一个变换矩阵#是一种坐标系变换
    
    eigValInd = argsort(eigVals)#排序,并获取排序后的下标 #sort, sort goes smallest to largest  #排序，将特征值按从小到大排列
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions      #选择维度为topNfeat的特征值
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest   #选择与特征值对应的特征向量
    print ("特征向量排序结果\n",redEigVects)#mat
    
    lowDDataMat = meanRemoved * redEigVects#(obj1-mean)*mat   #transform data into new dimensions    #将数据映射到新的维度上，lowDDataMat为降维后的数据
    print ("降维后的数据\n",lowDDataMat)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals          #对原始数据重构，用于测试
    print ("对原始数据重构\n",reconMat)#(obj1-mean)*mat*mat'+mean #相对位置不变，重构结果相同
    return redEigVects

'''
obj1=[[0,0,0],[1,0,0],[2,0,0]];
obj2=[[0,0,0],[0,1,0],[0,2,0]];
pca(obj1,3)
pca(obj2,3)
'''

covMat = cov([[1,1],[0,0],[-1,-1]], rowvar=0) 
eigVals,eigVects = linalg.eig(mat(covMat)) 
print(eigVals)
print(eigVects)
