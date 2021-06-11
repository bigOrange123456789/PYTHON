# -*- coding: utf-8 -*-
import SimpleITK as sitk
from sklearn import mixture
import numpy as np
import cv2 

imageT1 ="../1.png"
image = sitk.ReadImage(imageT1)
image_data = sitk.GetArrayFromImage(image)

train_data = image_data
gmm = mixture.GaussianMixture(n_components=3, covariance_type='diag', tol=0.01, max_iter=100, n_init=1, init_params='kmeans')
flatData = train_data.flatten()
gmm.fit(flatData[:,np.newaxis]) 
label_data = gmm.predict(flatData[:,np.newaxis]).reshape(train_data.shape) 

#https://blog.csdn.net/m0_37477175/article/details/103294145
print(label_data)
cv2.imshow('c',label_data)