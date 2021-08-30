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

def divide0(pic):
    w=pic.shape[0]
    h=pic.shape[1]
    result=np.ones([100,w/10,h/10])
    i1=0
    while i1<w/10:
        i2=0
        while i2<h/10:
            j1=0
            while j1<10:
                j2=0
                while j2<10:
                    result[j1*10+j2,ii,i2]=data[j1*w/10+i1][j2*w/10+i2]
                    j2=j2+1
                j1=j1+1
            i2=i2+1
        i1=i1+1
    return result

def divide(img,K):  #K是横向和纵向的块数
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
import numpy as np
v=divide(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]),2)
print(v)
#def test():
