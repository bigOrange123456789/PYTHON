import pandas as pd
import numpy as np
def mul():
    data= pd.read_csv('score.csv')#读
    data=data*1.5
    data_i=data.astype(int)
    v=data_i.values;
    l=data_i.values.shape[0]
    for i in np.arange(l):#1:data_i.values.shape(0):
        if data.values[i]-data_i.values[i]>0:
            data_i.values[i]=data_i.values[i]+1;     
    data_i.to_csv('score2.csv')

#mul()

data= pd.read_csv('data.csv')#读
print(data)
  
'''
data=pd.Series([0,1,2])
data=pd.Series([0,1],index=["a","b"])

#0.创建
data = pd.DataFrame(np.array([[1,2]]))#np转pd
data = pd.DataFrame([[1,2,3],[4,5,6],[0,0,0]],columns=['a','b','c'])#列标签
data2 = pd.DataFrame({"a":[1,2],"b":[3,4]})#行标签

#1.增
#行
data0=pd.concat([data,data],axis=0)#外连接
data0=pd.merge(data,pd.DataFrame([[4,5,6]],columns=['a','b','c']))#内连接
data0=pd.merge(data,pd.DataFrame([[7,8,9]],columns=['a','b','c']),how='outer')#外连接
#列
data0=pd.concat([data,data],axis=1)

#2.删
data0=data.drop(labels = ["a"],axis = 1) #删除列
data0=data.drop(index=[0,1]) #删除行

#3.改
data0=data*10+1;
data.at[0,"a"]=100;
data.at[[0,1],"a"]=100;
data.at[0]="_";#行
data.at[0:2,"a"]="a";#列
print(data)

#4.查
#列
data0=data["a"]
data0=data[["a","b"]]
#行
data0=data.loc[0]
data0=data.loc[[0,1]]
data0=pd.DataFrame(data, index = [1, 2])

#5.存储
data.to_csv('test.csv')#写
data= pd.read_csv('test.csv')#读

#6.数据处理
data.value_counts()#条形图统计
data.values#pd转np #numpy.ndarray #旧版data.as_matrix()

data.columns#列标签
set(data)#获取列标签并去重
'''