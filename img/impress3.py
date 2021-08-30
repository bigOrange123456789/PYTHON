import keras
from keras import layers
from keras.datasets import mnist
import numpy as np

import cv2
image = cv2.imread('test3.jpg')
print(type(image))
print(image.shape)
r=image[:,:,0]
g=image[:,:,1]
b=image[:,:,2]
(x_train, _), (x_test, _) = mnist.load_data()
x_train=np.array([r,g,b])
print(type(x_train))
print(x_train.shape)
wh=x_train.shape[2]
#数据处理
#(x_train, _), (x_test, _) = mnist.load_data()
#x_train=x_train[0:2,:,:]
print(type(x_train))
print(x_train.shape)
x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# This is the size of our encoded representations

data_dim=wh*wh

encoding_dim = wh  #编码维度 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = keras.Input(shape=(data_dim,))#This is our input image
#编码层
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)#"encoded" is the encoded representation of the input
#编码、解码
decoded = layers.Dense(data_dim, activation='sigmoid')(encoded)#"decoded" is the lossy reconstruction of the input

#解码模型
autoencoder = keras.Model(input_img, decoded)# This model maps an input to its reconstruction
#将我们的模型配置为使用每像素二元交叉熵损失和 Adam 优化器
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,#拟合
                epochs=50,
                batch_size=3,
                shuffle=True,
                validation_data=(x_train, x_train))

encoded_input = keras.Input(shape=(encoding_dim,))# This is our encoded (32-dimensional) input
decoder_layer = autoencoder.layers[-1]# Retrieve the last layer of the autoencoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))# Create the decoder model

#使用
encoder= keras.Model(input_img, encoded)#784
encoded_imgs = encoder.predict(x_train)#32# 进行编码/解码测试 #Encode and decode some digits
decoded_imgs = decoder.predict(encoded_imgs)#784# Note that we take them from the *test* set

#图片效果展示 # Use Matplotlib (don't ask)
import matplotlib.pyplot as plt
n = 3  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):# Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(wh, wh))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(wh, wh))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

