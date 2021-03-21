from numpy import *
       
class MaxTest: 
   def __init__(self):
      self.a1=mat([[1,0],[0,2]])
      self.a2=mat([[1,1],[0,2]])
   
   def test1(self):
     a1=self.a1
     a2=self.a2
     print(a1*a2)#乘法
     print(multiply(a1,a2))#点乘
     print("数乘",10*a2+100)#数乘
     print(a1.I)#求逆
     print(a2.T)#求转置
   def test2(self):
     a1=self.a1
     a2=self.a2
     a3=mat(ones(shape(a1)))#生成矩阵
     a3[0,0]=9#修改元素
     print(a3[0,0])#查询元素
   def test3(self):
       a1=self.a1
       a2=self.a2
       print("\n列和",a2.sum(axis=0) )#列和
       print(a2.sum(axis=1) )#行和
       print(a2.sum())#所有元素的和
   def test4(self):#元素和与最值
       print("列和",self.a2.sum(axis=0) )#列和
       print("行和",self.a2.sum(axis=1) )#行和
       print("元素和",self.a2.sum())#所有元素的和
       print("最大值",self.a2.max())
       print("最小值",self.a2.min())
   def test5(self):
       a1=self.a1
       a2=self.a2       
       print("\n复制：",a1.copy(),"\n复制：",copy(a1))
       for i in range(shape(a1)[0]):
           for j in range(shape(a1)[1]):
               print(a1[i,j])#遍历矩阵
 
m=MaxTest()
m.test4()
