import torch
import numpy as np


def f(x):
    return (x+1)**2

def computeGrad(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    for i in range(len(x)):
        x0=x[i]
        grad[i]=(f(x0)-f(x0-h))/h
    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=2000):
    x=init_x
    for i in range(step_num):
        g = computeGrad(f, x)
        x = x - g * lr
    return x

#g=computeGrad(f,[1,2])
#print("g",g,sep=":")
#test=[1,2]
#print(len(test))
x=gradient_descent(f,[10,100,-10])
print(x)