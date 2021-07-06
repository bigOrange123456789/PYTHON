import torch.nn as nn
import torch
from torch import autograd
from torchsummary import summary
 
class DoubleConv(nn.Module):
 def __init__(self, in_ch, out_ch):
  super(DoubleConv, self).__init__()
  self.conv = nn.Sequential(
   nn.Conv2d(in_ch, out_ch, 3, padding=0),
   nn.BatchNorm2d(out_ch),
   nn.ReLU(inplace=True),
   nn.Conv2d(out_ch, out_ch, 3, padding=0),
   nn.BatchNorm2d(out_ch),
   nn.ReLU(inplace=True)
  )
 
 def forward(self, input):
  return self.conv(input)
 
class Unet(nn.Module):
 def __init__(self, in_ch, out_ch):
  super(Unet, self).__init__()
  self.conv1 = DoubleConv(in_ch, 64)
  self.pool1 = nn.MaxPool2d(2)
  self.conv2 = DoubleConv(64, 128)
  self.pool2 = nn.MaxPool2d(2)
  self.conv3 = DoubleConv(128, 256)
  self.pool3 = nn.MaxPool2d(2)
  self.conv4 = DoubleConv(256, 512)
  self.pool4 = nn.MaxPool2d(2)
  self.conv5 = DoubleConv(512, 1024)
  # 逆卷积，也可以使用上采样
  self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
  self.conv6 = DoubleConv(1024, 512)
  self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
  self.conv7 = DoubleConv(512, 256)
  self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
  self.conv8 = DoubleConv(256, 128)
  self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
  self.conv9 = DoubleConv(128, 64)
  self.conv10 = nn.Conv2d(64, out_ch, 1)
 
 def forward(self, x):
  c1 = self.conv1(x)
  crop1 = c1[:,:,88:480,88:480]
  p1 = self.pool1(c1)
  c2 = self.conv2(p1)
  crop2 = c2[:,:,40:240,40:240]
  p2 = self.pool2(c2)
  c3 = self.conv3(p2)
  crop3 = c3[:,:,16:120,16:120]
  p3 = self.pool3(c3)
  c4 = self.conv4(p3)
  crop4 = c4[:,:,4:60,4:60]
  p4 = self.pool4(c4)
  c5 = self.conv5(p4)
  up_6 = self.up6(c5)
  merge6 = torch.cat([up_6, crop4], dim=1)
  c6 = self.conv6(merge6)
  up_7 = self.up7(c6)
  merge7 = torch.cat([up_7, crop3], dim=1)
  c7 = self.conv7(merge7)
  up_8 = self.up8(c7)
  merge8 = torch.cat([up_8, crop2], dim=1)
  c8 = self.conv8(merge8)
  up_9 = self.up9(c8)
  merge9 = torch.cat([up_9, crop1], dim=1)
  c9 = self.conv9(merge9)
  c10 = self.conv10(c9)
  out = nn.Sigmoid()(c10)
  return out
 
if __name__=="__main__":
 test_input=torch.rand(1, 1, 572, 572)
 model=Unet(in_ch=1, out_ch=2)
 summary(model, (1,572,572))
 ouput=model(test_input)
 print(ouput.size())