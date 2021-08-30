#开始数据准备
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
#.set(style='white', context='notebook', palette='deep')

# Load the data
train = pd.read_csv("train.csv")
train=train.drop(index=range(37000+4999))

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)# Drop 'label' column

x_train = (X_train/255.).values.astype('float32') / 255.
y_train = to_categorical(Y_train, num_classes = 10)# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
'''
#完成数据准备

#1.使用一个全连接的神经层作为编码器和解码器：
import keras
from keras import layers
encoding_dim = 32 # This is the size of our encoded representations # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = keras.Input(shape=(784,))# This is our input image
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)# "encoded" is the encoded representation of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)# "decoded" is the lossy reconstruction of the input

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

#2.创建一个单独的编码器模型：
encoder = keras.Model(input_img, encoded)# This model maps an input to its encoded representation

#3.以及解码器模型：
encoded_input = keras.Input(shape=(encoding_dim,))# This is our encoded (32-dimensional) input
decoder_layer = autoencoder.layers[-1]# Retrieve the last layer of the autoencoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))# Create the decoder model

#4.将我们的模型配置为使用每像素二元交叉熵损失和 Adam 优化器：
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_train, x_train))
# Encode and decode some digits     # Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
