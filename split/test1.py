import numpy as np
import cv2  

# 用于平滑灰度直方图
def smooth1d(line, k_size=11, n=1):
    for i in range(n):
        line = np.convolve(line, np.ones((k_size,))/k_size, mode='same')
    return line

# 寻找峰值，ignore用于忽略部分像素，多峰情况
def findPeak(line, ignore=100):
    peak = []
    trough = []
    for i in range(ignore, len(line) - 1):
        if line[i] == 0:
            continue
        if line[i] > 0 and line[i+1] < 0:
            peak.append(i)
        elif line[i] < 0 and line[i+1] > 0:
            trough.append(i)

    return np.array(peak), np.array(trough)

def twoMode(img_gray):
    hist, bins = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
    hist = smooth1d(hist, n=3)
    s = np.sign(np.diff(np.insert(hist, 0, values=0)))
    peak, trough = findPeak(s)
    peak = peak[np.argsort(hist[peak])[-2:]]
    trough = np.clip(trough, a_min=np.min(peak), a_max=np.max(peak))
    trough = trough[np.argmin(hist[trough])]

    print('2-mode:', trough)

    return trough

image = cv2.imread('../1.png')
t=twoMode(image)
print(t)

for i in range(len(image)):
    a=image[i];
    for j in range(len(a)):
        b=a[j];
        for k in range(len(b)):
            c=b[k];
            if(image[i,j,k]<t):
                image[i,j,k]=0;
            
cv2.imshow('cat',image)

