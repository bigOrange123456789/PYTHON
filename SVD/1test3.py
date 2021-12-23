#14-1
#三种计算向量相似度的方法
import numpy as np
#输入两个列向量
def euclidSim(inA,inB):#欧氏距离
    return 1./(1.+np.linalg.norm(inA-inB))

def pearsSim(inA,inB):#皮尔逊相关系数
    if len(inA)<3:
        return 1.
    else:
        return 0.5+0.5*np.corrocoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):#余弦相似度
    num=float(inA.T*inB)
    denom=np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)

