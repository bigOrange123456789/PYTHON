import cv2
import numpy as np
def arr(a,b):
    result=np.arange(b-a+1)+a
    return result
def rectangle(img,x1,y1,x2,y2):
    img1=img[arr(x1,x2),:,:]
    img2=img1[:,arr(y1,y2),:]
    return img2
image2=rectangle(image,300,300,600,600)
image = cv2.imread('../1.png')
print(image.shape)
image2=rectangle(image,300,300,600,600)
#300,300,600,600

cv2.imshow('cat',image2)
key=cv2.waitKey()
#cv2.imwrite('../test.png',image2)
