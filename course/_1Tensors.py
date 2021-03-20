from __future__ import print_function
import torch

#Tensors (张量) 个人认为张量就是矩阵

x = torch.empty(5, 3)#生成0矩阵
print(x)

x = torch.rand(5, 3)#生成随机矩阵
print(x)


x = torch.tensor([5.5, 3])
print(x)
print(x.size())