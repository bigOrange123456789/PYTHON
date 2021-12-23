#用python实现SVD
from numpy import *
a=[
    [1,1],
    [7,7]
    ]
b=[
   [1,1,1,0,0],
   [2,2,2,0,0],
   [1,1,1,0,0],
   [5,5,5,0,0],
   [1,1,0,2,2],
   [0,0,0,3,3],
   [0,0,0,1,1]
   ]
print(a)

u,s,vt = linalg.svd(a)
def getS(s0):
    n=len(s0)
    s1=eye(n)
    for i in range(n):
        s1[i][i]=s0[i]
    return s1
s3=getS(s)


s2 = zeros([2,2])
for i in range(2):
    s2[i][i] = s[i]

tmp = dot(u,s3)
tmp = dot(tmp,vt)

print(tmp)


