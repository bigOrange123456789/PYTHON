import pandas as pd
import numpy as np
import torch
from torchvision.utils import make_grid
def showPic(data):
    n_pic=data.shape[0]
    n_pixels=data.shape[1]
    width=int(n_pixels**0.5)
    pwh=data.reshape((n_pic, width, width))#数据重组#(n, 784)->(n, 28, 28)
    pcwh=torch.Tensor(pwh).unsqueeze(1)#在第一层的位置添加只有1个元素的维度
    cwh = make_grid(pcwh)#将多张图片合并，并且生成rgb
    whc=cwh.numpy().transpose((1,2,0))#(高，宽，通道)(32, 242, 3)

    import cv2
    cv2.imshow('img0',whc)
    cv2.waitKey(0)
    cv2.imwrite('img0.jpg',whc)

train_df = pd.read_csv('./test.csv')
data=train_df.values[[1,2],0:]#要显示的图片行号,第1列到最后一列
showPic(data)
