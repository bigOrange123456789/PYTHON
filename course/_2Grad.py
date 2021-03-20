import torch
import numpy as np

x = torch.ones(2, 2, requires_grad=True)#一矩阵
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()#向后的

print(x)
print(y)
print(z)
print(out)
print(x.grad)