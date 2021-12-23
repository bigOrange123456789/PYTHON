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
#############基于SVD的评分估计
def svdEst(dataMat,user,simMeas,item):
    #      数据矩阵 用户 sim方法 物品编号
    n=np.shape(dataMat)[1] #列数-货物个数  
    simTotal=0 
    ratSimTotal=0 
    
    #add1
    U,Sigma,VT=np.linalg.svd(dataMat)
    Sig4=np.mat(np.eye(4)*Sigma[:4])
    xformedItems=dataMat.T*U[:,:4]*Sig4.I
    
    for j in range(n):#遍历所有货物
         userRating=dataMat[user,j]#查看用户对该货物的评价
         if userRating==0 or j==item:#用户没有评价过这个货物 或 j就是这个货物
             continue
         
         #add2
         similarity=simMeas(
             xformedItems[item,:].T,
             xformedItems[j   ,:].T,
             )
         '''
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
         '''
        
         simTotal+=similarity
         ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal#基于相似度的得分加权和

