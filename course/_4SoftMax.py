import torch
import numpy as np
import math

def softmax(x):
    s = np.zeros_like(x)
    sum=0
    for i in range(len(x)):
        s[i]=math.exp(x[i])
        sum+=s[i]
    return s/sum

print(softmax([0.3,2.9,4.0]))