from numpy import *
def add_row(mat,arr):
    mat=mat.tolist();
    return array(mat+[arr])
def add_col(mat,arr):
    #print("mat.T",mat.T)
    mat=(mat.T).tolist();
    return array(mat+[arr]).T
def pca(dataMat, topNfeat):    #topNfeat 降维后的维度
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
    
    a=1;
    b=1;
    c=1;
    if meanVals[0]<0:
        a=-1;
    if meanVals[1]<0:
        b=-1;   
    if meanVals[2]<0:
        c=-1;  
    redEigVects=redEigVects*[
        [a,0,0],
        [0,b,0],
        [0,0,c]
    ]  
    normalization= meanRemoved * redEigVects
    return redEigVects,meanVals,normalization

def f(obj1,obj2):
    m1,mean1,normal1 =pca(obj1,topNfeat=3)#obj0=(obj1-mean)*mat*mat' 
    print("归一化结果\n",normal1)#降维后的数据#归一化分布，但不归一化次序
    m2,mean2,normal2 =pca(obj2,topNfeat=3)#obj0=obj2*mat'+mean2
    print("归一化结果\n",normal2)#降维后的数据#归一化分布，但不归一化次序
    m=m1*m2.I
    obj=(obj1-mean1)*m+mean2
    print("结果验证：\n",obj)


obj1=[[0,0,0],[2,0,0],[2,1,0]];
obj2=[[0,0,1],[2,0,1],[2,1,1]];
obj3=[[0,0,0],[0,2,0],[-1,2,0]];

f(obj1,obj2)

