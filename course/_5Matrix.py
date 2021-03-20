from numpy import *

a1=mat([
    [1,0],
    [0,2]
       ])
a2=mat([
    [1,1],
    [0,2]
       ])

print(a1*a2)#乘法
print(multiply(a1,a2))#点乘
print(10*a2)#数乘
print(a1.I)#求逆
print(a2.T)#求转置

a3=mat(ones(shape(a1)))#生成矩阵
a3[0,0]=9#修改元素
print(a3[0,0])#查询元素

print(a2.sum(axis=0) )#列和
print(a2.sum(axis=1) )#行和
print(a2.sum())#所有元素的和

for i in range(shape(a1)[0]):
    for j in range(shape(a1)[1]):
        print(a1[i,j])#遍历矩阵