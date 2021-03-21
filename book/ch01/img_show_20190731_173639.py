# coding: utf-8
import matplotlib.pyplot as plt #导入一个类，并 重命名
from matplotlib.image import imread #导入一个类中的函数

img = imread('../dataset/lena.png') #读入图像 读入图
plt.imshow(img)#绘制图

plt.show()#展示绘制的图片