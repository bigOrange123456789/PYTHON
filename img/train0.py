import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import cv2
image = cv2.imread('test.jpg')
wh=image.shape[0]
image= cv2.resize(image,dsize=(wh,wh))

r=image[:,:,0]
g=image[:,:,1]
b=image[:,:,2]
x_train=np.array([r,g,b])

#数据处理
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

data_dim=wh*wh
encoding_dim = wh  #编码维度 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = keras.Input(shape=(data_dim,))#This is our input image

encoded = layers.Dense(encoding_dim, activation='relu')(input_img)#编码层 #"encoded" is the encoded representation of the input
decoded = layers.Dense(data_dim, activation='sigmoid')(encoded)#编码、解码 #"decoded" is the lossy reconstruction of the input

#解码模型
autoencoder = keras.Model(input_img, decoded)# This model maps an input to its reconstruction
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')#每像素二元交叉熵损失和 Adam 优化器

autoencoder.fit(x_train, x_train,#拟合
                epochs=25,
                batch_size=10,
                shuffle=True,
                validation_data=(x_train, x_train))
encoded_input = keras.Input(shape=(encoding_dim,))# This is our encoded (32-dimensional) input
autoencoder.save( filepath="modelResult.h5", overwrite=True, include_optimizer=True )
