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
class Data():
    def __init__(self,train_x,train_y):
        # training data
        self.X = train_x
        self.y = torch.from_numpy(train_y)
        self.transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,),std=(0.5,))#均值，方差
            ])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]
data=np.load('data.npz')
train_loader = torch.utils.data.DataLoader(dataset=Data(data['train_x'],data['train_y']),
                                           batch_size=64, shuffle=True)
#1.定义
#1.1 定义模型
from net0 import Net#导入模型
model = Net()

#1.2定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()#选择误差计算方法
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#2.训练
#2.1 拟合
def train(epoch):
    model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

#2.2 评价
def evaluate(data_loader):
    model.eval()
    correct = 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print("准确率为：",str(100. * correct.numpy() / len(data_loader.dataset))+"%")
n_epochs = 1
for epoch in range(n_epochs):
    train(epoch)
    evaluate(train_loader)
torch.save(model.state_dict(), "modelResult.pkl")
