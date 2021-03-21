# coding: utf-8
import numpy as np

#用神经网络实现了一个或门
def AND(x1, x2):
    x = np.array([x1, x2])#将python自带的数组类型转为了np的数组类型
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b#这是一个激活函数
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
