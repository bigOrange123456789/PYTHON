import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
class Data():
    def __init__(self,data0):
        self.X = data0.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean=(0.5,), std=(0.5,))])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.transform(self.X[idx])
inputData=pd.read_csv('../input/test.csv').loc[0:10].values
test_loader = DataLoader(dataset=Data(inputData), shuffle=False)#shuffle洗牌 打乱数据

from net0 import Net#导入模型
model = Net()
model.load_state_dict(torch.load("modelResult.pkl"))
#进行预测
model.eval()
result=[]
for _, data in enumerate(test_loader):
     pic = Variable(data, volatile=True)
     pred = model(pic).data.max(1, keepdim=True)[1]
     result.append(pred.numpy()[0][0])
print(result)

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
showPic(inputData)
