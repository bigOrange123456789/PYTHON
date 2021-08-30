import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import cv2
#数据读取
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

input_img = keras.Input(shape=(data_dim,))#This is our input image#解码模型
encoded_input = keras.Input(shape=(encoding_dim,))# This is our encoded (32-dimensional) input
m=keras.models.load_model( "modelResult.h5" ).layers
encoder= keras.Model(input_img, m[1](input_img))#编码模型  # input_img 似乎没有参数#784
decoder = keras.Model(encoded_input, m[2](encoded_input))# Create the decoder model

encoded_imgs = encoder.predict(x_train)#32# 进行编码/解码测试 #Encode and decode some digits
decoded_imgs = decoder.predict(encoded_imgs)#784# Note that we take them from the *test* set
#https://blog.csdn.net/zhw864680355/article/details/103405677

rgb=decoded_imgs.reshape(3,wh, wh)
result=255.*rgb.transpose((1,2,0))#getImg(rgb,wh,wh)
cv2.imwrite('test2.jpg',result)

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
