# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

#sigmoid函数的实现 用来模拟跳跃函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
