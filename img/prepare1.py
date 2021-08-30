import numpy as np
import cv2
def divide(img,K):  #每一块的大小
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

image = cv2.imread('test.jpg')
image= cv2.resize(image,dsize=(image.shape[0],image.shape[0]))

r=image[:,:,0]
g=image[:,:,1]
b=image[:,:,2]
print(r.shape)
wh=10
r2=divide(r,wh)
g2=divide(g,wh)
b2=divide(b,wh)

data=np.ones((r2.shape[0]*3,r2.shape[1]))
for i in range(r2.shape[0]):
    data[i,:]=r2[i,:]
    data[i+r2.shape[0],:]=g2[i,:]
    data[i+r2.shape[0]*2,:]=b2[i,:]

print(data.shape)
#x_train=np.array([r,g,b])
x_train = data.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
np.savez('x_train',x_train=x_train,wh=wh,l=image.shape[0]/wh)
print("x_train.shape",x_train.shape)
print("wh",wh)
print("l",image.shape[0]/wh)


