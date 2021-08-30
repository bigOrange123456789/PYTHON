import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

np.random.seed(2)
sns.set(style='white', context='notebook', palette='deep')

# Load the data
train = pd.read_csv("../input/train.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)

# free some space
del train

Y_train.value_counts()

# Normalize the data
X_train = X_train / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


import keras
model = keras.models.load_model( "modelResult.h5" )
#3.检验
#3.1测试集
test = pd.read_csv("../input/test.csv")
test=test.loc[0:2]
import pandas as pd
import numpy as np
import torch
from torchvision.utils import make_grid
import cv2
def showPic(data):
    n_pic=data.shape[0]
    n_pixels=data.shape[1]
    width=int(n_pixels**0.5)
    pwh=data.reshape((n_pic, width, width))#数据重组#(n, 784)->(n, 28, 28)
    pcwh=torch.Tensor(pwh).unsqueeze(1)#在第一层的位置添加只有1个元素的维度
    cwh = make_grid(pcwh)#将多张图片合并，并且生成rgb
    whc=cwh.numpy().transpose((1,2,0))#(高，宽，通道)(32, 242, 3)

    cv2.imshow('img0',whc)
    cv2.waitKey(0)
    #cv2.imwrite('img0.jpg',whc)
showPic(test.values)
test = test / 255.0
test = test.values.reshape(-1,28,28,1)

# Predict the values from the validation dataset
pred = model.predict(test)
# Convert predictions classes to one hot vectors
pred_classes = np.argmax(pred,axis = 1)
print(pred_classes)
