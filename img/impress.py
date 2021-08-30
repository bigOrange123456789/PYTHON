from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
import keras

#图像切割
def divide(img,K):
  col = K*K
  nrows,ncols = img.shape
  R = int(nrows/K)
  C = int(ncols/K)
  vec = np.ones((R*C,col))
  row = 0
  for i in range(R):
    for j in range(C):
      vec[row,0:col] = img[i*K:(i+1)*K,j*K:(j+1)*K].reshape(1,col)
      row += 1
  return vec

#利用keras，作为稀疏编码转换，传入参数vec是divide后的图片向量
def autoencoder(vec):#自动编码器
  nrows, ncols = vec.shape
  input_dim = ncols
  encoding_dim = int(input_dim*0.8) #设置隐藏层的维度，压缩比为1:0.8
  input_img = Input(shape=(100,))
  encoded = Dense(encoding_dim, activation='relu')(input_img)
  decoded = Dense(100, activation='sigmoid')(encoded)
  #autoencoder = Model(input=input_img, output=decoded)
  autoencoder = keras.Model(input_img, decoded)

  #encoder = Model(input=input_img, output=encoded)
  encoder = keras.Model(input_img, encoded)
  encoded_input = Input(shape=(encoding_dim,))
  decoder_layer = autoencoder.layers[-1]
  #decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
  decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
  autoencoder.fit(vec, vec, epochs=80, batch_size=256, shuffle=True, validation_data=(vec, vec))
  encoded_imgs = encoder.predict(vec)
  decoded_imgs = decoder.predict(encoded_imgs)
  return decoded_imgs


#图像切割后，逆转程序
def redivide(div_img,K,img):
  rows,cols = img.shape
  #nrows,ncols = div_img.shape
  R = int(rows/K)
  C = int(cols/K)
  vec = np.ones((rows,cols))
  count = 0
  for i in range(R):
    for j in range(C):
      vec[i*K:(i+1)*K, j*K:(j+1)*K] = div_img[count].reshape(K,K)
      count += 1
  return vec

def normalize_data(data):  # 0.1<=data[i][j]<=0.9
    data = data - np.mean(data)
    pstd = 3 * np.std(data)
    data = np.maximum(np.minimum(data, pstd), -pstd) / pstd
    data = (data + 1.0) * 0.4 + 0.1
    return data

def plotdivide(img,K):
  nrows, ncols = img.shape
  rows = nrows/K
  cols = ncols/K
  vec = divide(img, K)
  img_count = int(rows*cols)
  for i in range(img_count):
    ax = plt.subplot(rows, cols, i+1)
    plt.imshow(vec[i].reshape(K,K))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.gray()
  plt.show()

if __name__ == "__main__":
  #切割图片验证，和切割后逆转验证。原始是4x4 ， 按2x2切割
  # mat = np.array([[1,2,3,4],[5,6,7,8],[2,4,6,8],[11,12,13,14]])
  # vec = divide(mat, 2)
  # print(vec)
  # reimg = redivide(vec,2,mat)
  # print("原始 \n",mat)
  # print("redivide is \n",reimg)
  # print(reimg)

  #图像切分，并做归一化
  #img = np.array(Image.open('D:\\pythoncode\\learn1\\lesson1\\zhuozi2.jpeg').convert('L'))
  #normalize_img = normalize_data(img)
  #print(normalize_img)

  #切割图片
  #img = np.array(Image.open('D:\\pythoncode\\learn1\\lesson1\\zhuozi2.jpeg').convert('L'))
  #normalize_img = normalize_data(img)
  #vec = divide(normalize_img, 10)
  #print(vec)

  #切割图片后，逆向操作，显示图片
  #img = np.array(Image.open('D:\\pythoncode\\learn1\\lesson1\\zhuozi2.jpeg').convert('L'))
  #normalize_img = normalize_data(img)
  #vec = divide(normalize_img, 10)
  #reimg = redivide(vec,10,img)
  #plt.imshow(img, cmap='gray')
  #plt.show()

  #切割图片后，显示切割的图片
  '''
  img = np.array(Image.open('test.jpg').convert('L'))#读取图片
  print(img.shape)
  print(np.array(Image.open('test.jpg')).shape)
  arr=np.array(Image.open('test.jpg'))
  r=arr[:,:,0]
  g=arr[:,:,1]
  b=arr[:,:,2]
  i1=0
  i2=0
  i3=0
  result=[]
  while i1<500:
    result.append([])
    while i2<500:
        result[i1].append([])
        result[i1][i2].append(r[i1,i2])
        result[i1][i2].append(g[i1,i2])
        result[i1][i2].append(b[i1,i2])
        i2=i2+1
    i1=i1+1
  '''
  '''
  normalize_img = normalize_data(img)#数据标准化
  vec = divide(normalize_img, 10)#图片切分？
  img_encoder = autoencoder(vec)#自动编码
  reimg = redivide(img_encoder,10,img)
  #plotdivide(reimg, 10)  #查看切割后，图片每个区域样子
  plt.imshow(reimg, cmap='gray')
  plt.show()
  '''
