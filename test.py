import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset 
#datas = pd.read_csv('data.csv') 
#datas 

from sklearn.linear_model import LinearRegression 

x=[1,2,3]
y=[1,2,3]

lin = LinearRegression() 
lin.fit(x, y) 
print(x,y)