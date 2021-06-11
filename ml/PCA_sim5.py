from numpy import *
import json

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

def getMatrix(obj1,obj2):
    m1,mean1,normal1 =pca(obj1["arr"],topNfeat=3)#obj0=(obj1-mean)*mat*mat' 
    print("归一化结果\n",normal1)#降维后的数据#归一化分布，但不归一化次序
    m2,mean2,normal2 =pca(obj2["arr"],topNfeat=3)#obj0=obj2*mat'+mean2
    print("归一化结果\n",normal2)#降维后的数据#归一化分布，但不归一化次序
    m=m1*m2.I
    center1=obj1["center"]
    center2=obj2["center"]
    y=mean2-dot(mean1,m)+array(center2)-array(center1)
    
    m=array(m.tolist()+y.tolist())
    m=array(m.T.tolist()+[[0,0,0,1]]).T
    print("仿射变换矩阵为：\n",m)    
    return m
def read(url):
    with open(url, 'r') as f:
        j = json.load(f)    #此时a是一个字典对象
        return j;
def save(mat,name):
    arr=[]
    for i in mat:
        for j in i:
            arr.append(j)
    with open(name, 'w') as f:
        json.dump({"m":arr},f)


obj1=read("mesh00.json");#[[0,0,0],[2,0,0],[2,1,0]];
obj2=read("mesh01.json");#[[0,10,1],[2,10,1],[2,11,1]];
m=getMatrix(obj1,obj2)
save(m,"matrix.json")