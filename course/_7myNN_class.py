from numpy import *
import math

#0.定义工具函数


#1.定义xtw
x=mat([0.6,0.9])
w=mat(ones((2,3)))
t=mat([0,0,1])

#2. x*w求y
y=x*w

#3.通过softmax函数处理y
def softmax(a):#x是一个矩阵
    b=copy(a)
    b=b-b.max()
    for i in range(shape(a)[0]):
        for j in range(shape(a)[1]):
            b[i,j]=math.exp(a[i,j]) # 遍历矩阵
    return b/sum(b)
#y_s=softmax(y)

#4.求损失函数
def MSE(y,t):#均方误差 y,t均为矩阵
    b = mat(ones(shape(y)))
    for i in range(shape(b)[0]):
        for j in range(shape(b)[1]):
            y0=y[i,j]
            t0=t[i,j]
            b[i,j]=(y0-t0)**2/2 # 遍历矩阵
    return sum(b)
def CEE(y,t):#交叉熵shang误差,用于分类问题中 y是神经网络的输出,t是标签#cross entropy erro
    b = mat(ones(shape(y)))
    for i in range(shape(b)[0]):
        for j in range(shape(b)[1]):
            y0=y[i,j]
            t0=t[i,j]
            b[i,j]=t0*log(y0) # 遍历矩阵
    return -1*sum(b)
#loss=CEE(y_s,t)

#5.定义匿名函数，将w变为损失函数的参数 f=loss(w)
def f(w0):#x和t是已知的
    y0=x*w0
    y0_s=softmax(y0)
    loss0=CEE(y0_s,t)
    return loss0
#f(w)

#6.求损失函数关于w的梯度
def computeGrad(f,w):
    h=1e-4
    grad=mat(ones(shape(w)))
    #print(w)
    for i in range(shape(w)[0]):
        for j in range(shape(w)[1]):
            w2=copy(w)
            w2[i,j]=w[i,j]+h
            grad[i,j]=f(w2)-f(w)/h
    return grad
#grad=computeGrad(f,w)

#7.使用梯度下降的方法计算w
def gradient_descent(f,init_w,lr=0.01,step_num=5):
    W0=init_w
    for i in range(step_num):
        g = computeGrad(f, W0)
        W0 = W0 - g * lr
    return W0
W=gradient_descent(f,w)


print(x*W)

class simpleNet:
    def _init_(self):
        self.x=mat([0.6,0.9])
        self.w=mat(ones((2,3)))
        self.t=mat([0,0,1])
        
        








