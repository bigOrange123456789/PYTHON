import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import cv2
npzfile=np.load('x_train.npz')
x_train=npzfile['x_train']
wh=npzfile['wh']#int(npzfile['wh']/npzfile['partion'])
print(wh)

data_dim=wh*wh
encoding_dim = wh*25  #编码维度 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
print(wh)
input_img = keras.Input(shape=(data_dim,))#This is our input image

encoded = layers.Dense(encoding_dim, activation='relu')(input_img)#编码层 #"encoded" is the encoded representation of the input
decoded = layers.Dense(data_dim, activation='sigmoid')(encoded)#编码、解码 #"decoded" is the lossy reconstruction of the input

#解码模型
autoencoder = keras.Model(input_img, decoded)# This model maps an input to its reconstruction
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')#每像素二元交叉熵损失和 Adam 优化器
print(x_train.shape)
autoencoder.fit(x_train, x_train,#拟合
                epochs=60,#将数据使用多少遍
                batch_size=1,#将7500个数据分成多少份
                shuffle=True,
                validation_data=(x_train, x_train))
encoded_input = keras.Input(shape=(encoding_dim,))# This is our encoded (32-dimensional) input
autoencoder.save( filepath="modelResult.h5", overwrite=True, include_optimizer=True )
