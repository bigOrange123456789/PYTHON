import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers

import matplotlib.pyplot as plt

train_df = pd.read_csv('../input/train.csv')

n_train = len(train_df)
n_pixels = len(train_df.columns) - 1
n_class = len(set(train_df['label']))

random_sel = np.random.randint(n_train, size=8)#长度为8的随机整数矩阵

data=train_df.values[random_sel, 1:]#第1列到最后一列
data=data/255. # (8, 784)
pics=data.reshape((8, 28, 28))
tensor=torch.Tensor(pics)
#print(tensor)
#print(tensor.unsqueeze(1))
grid = make_grid(
    torch.Tensor(pics).unsqueeze(1), 
    nrow=8)


show=grid.numpy()#(通道，高，宽)一张图片 3个通道和宽高 #(3, 32, 242)
show=show.transpose((1,2,0))#(高，宽，通道)(32, 242, 3)
plt.imshow(show)
#plt.axis('off')#隐藏坐标轴