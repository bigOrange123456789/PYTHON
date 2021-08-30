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

#1.定义
#1.1定义模型
# Set the CNN model 设置CNN模型
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

#1.2定义优化器
# Define the optimizer 定义优化器
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model 编译模型
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer 设置学习率衰退器
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

#2训练
#2.1数据增强
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

#2.2拟合模型
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

model.save( filepath="modelResult.h5", overwrite=True, include_optimizer=True )
#filepath：保存的路径
#overwrite：如果存在源文件，是否覆盖
#include_optimizer：是否保存优化器状态
#ex : mymodel.save(filepath="p402/my_model.h5", includeoptimizer=False)

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
