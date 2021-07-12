import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

test = pd.read_csv("../input/test.csv")
test=test.loc[0:2]
def showPic(data):
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import torch
    data=data.values
    data=data/255. # (n, 784)
    n_pixels=len(data[0])
    pwh=data.reshape((len(data), int(n_pixels**0.5), int(n_pixels**0.5)))#两张图片，每张图片为28*28  #(n, 784)->(n, 28, 28)
    pcwh=torch.Tensor(pwh).unsqueeze(1)
    cwh = make_grid(pcwh)
    whc=cwh.numpy().transpose((1,2,0))#(高，宽，通道)(32, 242, 3)
    plt.imshow(whc)
showPic(test)
test = test / 255.0
test = test.values.reshape(-1,28,28,1)