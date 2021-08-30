import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

np.random.seed(2)
#.set(style='white', context='notebook', palette='deep')

# Load the data
train = pd.read_csv("../input/train.csv")
print(train.shape)
train=train.drop(index=range(37000))
print(train.shape)

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)# Drop 'label' column

X_train = (X_train/255.).values.reshape(-1,28,28,1)#Normalize& Reshape 3 dimensions (h= 28, w= 28 , canal = 1)
Y_train = to_categorical(Y_train, num_classes = 10)# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

#将数据差分为训练集和验证集 Split the train and the validation set for the fitting #验证validation
print(X_train.shape, Y_train.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
#test_size是测试数据在总数据中的比例 random_state是随机数种子
#X_train.shape 训练集输入, X_val.shape 测试集输入, Y_train.shape训练集输出, Y_val.shape 测试集输出

#1.定义
#1.1定义模型 Set the CNN model
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))#out 28 28 32
                 #kernel_size：卷积核大小
                 #padding = 'Same'：填充输入以使输出具有与原始输入相同的长度。尽可能两边添加同样数目的零列，如果要添加的列数为奇数个。那么让右边列的个数多一个即可。
                 #filters：输出空间的维度，滤波器的数量
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))#out 28 28 32
#model.add(MaxPool2D(pool_size=(2,2)))估计
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))#out 14 14 32
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))#out 14 14 64
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))#out 14 14 64
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))#out 7 7 64
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))#out 256 7 7 64
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

#1.2定义优化器
#优化器是编译模型的所需的两个核心参数之一（另一个是损失函数）
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#RMSProp算法加了一个衰减系数来控制历史信息的获取多少,rho与decay都是衰减因子
#lr是学习率,rho为吴恩达视频中得β，epsilon即公式中防止出现0，decay官方文档说明为每次更新学习率下降多少

# Compile the model 编译模型
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer 设置学习率衰退器
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

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

#2.2拟合模型Fit the model
batch_size = 43#86#344#172#86
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),#batch_size一次训练所抓取的数据样本数量
                              epochs = 1 ,# Turn epochs to 30 to get 0.9967 accuracy
                              validation_data = (X_val,Y_val),
                              verbose = 1,
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])
#epochs：整数，数据迭代的轮数
#validation_data：具有三种形式--生成验证集的生成器,（inputs,targets）,（inputs,targets，sample_weights）
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
#steps_per_epoch：当生成器一个epoch的执行次数

for i in history.history:
    print(i,history.history[i])
#loss，acc，val_loss，val_acc
#训练过程曲线分析：acc/loss/val_acc/val_loss
#验证集曲线震荡析原因：训练的batch_size太小
#分类问题一般看validation accuracy
#loss 是我们预先设定的损失函数计算得到的损失值；
#accuracy 是模型在数据集上基于给定 label 得到的评估结果

model.save( filepath="modelResult.h5", overwrite=True, include_optimizer=True )
#filepath：保存的路径     overwrite：如果存在源文件，是否覆盖      include_optimizer：是否保存优化器状态
