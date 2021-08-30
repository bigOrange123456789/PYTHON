import pandas as pd
import numpy as np
import keras
#读取数据
test = pd.read_csv("../input/test.csv")
test=test.loc[0:10]

#预测结果
model = keras.models.load_model( "modelResult.h5" )
test2 = (test / 255.).values.reshape(-1,28,28,1)#数据重组
pred = model.predict(test2)
pred_classes = np.argmax(pred,axis = 1)
print(pred_classes)

#验证预测
def showPic(data):
    import torch
    from torchvision.utils import make_grid
    import cv2
    n_pic=data.shape[0]
    n_pixels=data.shape[1]
    width=int(n_pixels**0.5)
    pwh=data.reshape((n_pic, width, width))#数据重组#(n, 784)->(n, 28, 28)
    pcwh=torch.Tensor(pwh).unsqueeze(1)#在第一层的位置添加只有1个元素的维度
    cwh = make_grid(pcwh)#将多张图片合并，并且生成rgb
    whc=cwh.numpy().transpose((1,2,0))#(高，宽，通道)(32, 242, 3)

    cv2.imshow('img0',whc)
    cv2.waitKey(0)
showPic(test.values)
