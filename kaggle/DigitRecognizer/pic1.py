import pandas as pd
import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def showPic(data):
    data=data/255. # (n, 784)
    n_pixels=len(data[0])
    pwh=data.reshape((2, int(n_pixels**0.5), int(n_pixels**0.5)))#两张图片，每张图片为28*28  #(n, 784)->(n, 28, 28)
    pcwh=torch.Tensor(pwh).unsqueeze(1)
    cwh = make_grid(pcwh)
    whc=cwh.numpy().transpose((1,2,0))#(高，宽，通道)(32, 242, 3)
    plt.imshow(whc)
train_df = pd.read_csv('../input/test.csv')
data=train_df.values[[1,2],0:]#要显示的图片行号,第1列到最后一列
showPic(data)# (n, 784)