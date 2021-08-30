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
np.savez('x_train',x_train=x_train,wh=wh)
