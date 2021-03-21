# coding: utf-8
import numpy as np #类numpy 是python中数字处理的工具
import matplotlib.pyplot as plt #矩阵绘图包中的绘图工具类

# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.show()