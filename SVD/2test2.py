import numpy as np
############计算相似度
def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)

#############推荐算法
A=[#goods0 goods1 goods2
   [1,1,1,0,0],#user0
   [2,2,2,0,0],#user1
   [1,1,1,0,0],#user2
   [5,5,5,0,0],#user3
   [1,1,0,2,2],
   [0,0,0,3,3],
   [0,0,0,1,1]
   ]
#用户对物体的估计评分
def standEst(dataMat,user,simMeas,item):
    #        数据矩阵 用户 sim方法 物品编号
    n=np.shape(dataMat)[1]#列数-货物个数  
    simTotal=0
    ratSimTotal=0
    for j in range(n):#遍历所有货物
         userRating=dataMat[user,j]#查看用户对该货物的评价
         if userRating==0:#用户没有评价过这个货物
             continue
         #变量overLap给出的是两个物品中已经被评分的用户
         overLap=np.nonzero(#返回数组中非零元素的索引值数组
             np.logical_and(#逻辑与
                 dataMat[:,item]>0,
                 dataMat[:,j]>0)
             )[0]
         
         if len(overLap)==0:#如果没有用户同时分析过这两个物品
             similarity=0
         else:
             similarity=simMeas(dataMat[overLap,item],
                                dataMat[overLap,j])#计算一下用户们对这两个物品评价的相关性
         simTotal+=similarity
         ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal#基于相似度的得分加权和
#推荐引擎
def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems=np.array(
        np.nonzero(dataMat[user,:]==0)#返回数组中元素的索引值数组
    )[0]
    if len(unratedItems)==0:
        return 'you rated everything'
    itemScores=[]
    for item in unratedItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]
    #sorted 排序

A=np.array(A)
print( recommend(A,5,N=3) )#只有一个对象未评价的时候会出错

