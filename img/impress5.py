import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import cv2
image = cv2.imread('test.jpg')
print("shape",image.shape)
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
print("wh:",wh)
encoding_dim = wh  #编码维度 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = keras.Input(shape=(data_dim,))#This is our input image
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)#编码层 #"encoded" is the encoded representation of the input
decoded = layers.Dense(data_dim, activation='sigmoid')(encoded)#编码、解码 #"decoded" is the lossy reconstruction of the input
print("decoded",type(decoded))#KerasTensor
#解码模型
autoencoder = keras.Model(input_img, decoded)# This model maps an input to its reconstruction
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')#每像素二元交叉熵损失和 Adam 优化器
print("shape",decoded.shape)
print("name",decoded.name)

for layer in autoencoder.layers:
   print(len(layer.weights))
   print(type(layer.weights))
   for weight in layer.weights:
        print(weight.name, weight.shape)

autoencoder.fit(x_train, x_train,#拟合
                epochs=25,
                batch_size=10,
                shuffle=True,
                validation_data=(x_train, x_train))
encoded_input = keras.Input(shape=(encoding_dim,))# This is our encoded (32-dimensional) input
m=autoencoder.layers

#获取训练好的模型的最后一层
decoder_layer = m[2]# Retrieve the last layer of the autoencoder model
#解码模型 用这一层的参数新建一个模型
decoder = keras.Model(encoded_input, m[2](encoded_input))# Create the decoder model
#                     k              kk-k   k

#编码模型  # input_img 似乎没有参数
encoder= keras.Model(input_img, m[1](input_img))#784

encoded_imgs = encoder.predict(x_train)#32# 进行编码/解码测试 #Encode and decode some digits
decoded_imgs = decoder.predict(encoded_imgs)#784# Note that we take them from the *test* set
#https://blog.csdn.net/zhw864680355/article/details/103405677

def getImg(rgb,w,h):
    data=rgb.reshape(3,w, h)
    result=np.ones([wh,wh,3])
    i1=0
    while i1<w:
        i2=0
        while i2<h:
            i3=0
            while i3<3:
                result[i1,i2,i3]=data[i3][i1][i2]*255
                i3=i3+1
            i2=i2+1
        i1=i1+1
    return result

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
