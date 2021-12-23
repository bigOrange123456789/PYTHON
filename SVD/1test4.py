#用python实现SVD
from numpy import *
b=[
   [1,1,1,0,0],
   [2,2,2,0,0],
   [1,1,1,0,0],
   [5,5,5,0,0],
   [1,1,0,2,2],
   [0,0,0,3,3],
   [0,0,0,1,1]
   ]
U,Sigma,VT=linalg.svd(b)
n=len(Sigma)#3
def getS(s0):
    #n=len(s0)
    s1=eye(n)
    for i in range(n):
        s1[i][i]=s0[i]
    return s1
s=getS(Sigma)

temp=dot(U[:,:n],s)
temp=dot(temp,VT[:n,:])
print(temp)
