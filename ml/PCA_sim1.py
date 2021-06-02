from numpy import *
def pca(dataMat, topNfeat=9999999):    #topNfeat 降维后的维度
    #去均值，将样本数据的中心点移到坐标原点
    meanVals = mean(dataMat, axis=0)#按列求均值，即每一列求一个均值，不同的列代表不同的特征 
    meanRemoved = dataMat - meanVals#  
    
    #计算协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)         
    
    #计算协方差矩阵的特征值和特征向量    
    #确保方差最大  ，构造新特征两两独立       
    eigVals,eigVects = linalg.eig(mat(covMat)) 
    eigValInd = argsort(eigVals)#排序,并获取排序后的下标 #sort, sort goes smallest to largest  #排序，将特征值按从小到大排列
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions      #选择维度为topNfeat的特征值
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest   #选择与特征值对应的特征向量
    normalization= meanRemoved * redEigVects
    return redEigVects,meanVals,normalization

obj1=[[0,0,0],[1,0,0],[2.5,0,0],[3,0,0]];
obj2=[[0,0,0],[-1,0,0],[-2.5,0,0],[-3,0,0]];

m1,mean1,normal1 =pca(obj1,topNfeat=3)#obj0=(obj1-mean)*mat*mat' 
print("归一化结果\n",normal1)#降维后的数据#归一化分布，但不归一化次序
m2,mean2,normal2 =pca(obj2,topNfeat=3)#obj0=obj2*mat'+mean2
print("归一化结果\n",normal2)#降维后的数据