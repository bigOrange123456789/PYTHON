import os
from scipy import io
from scipy.stats import norm
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

plot_dir = 'EM_out'
if os.path.exists(plot_dir) == 0:
    os.mkdir(plot_dir)
# 数据加载与解析
mask = io.loadmat('mask.mat')['Mask']                 # 数据为一个字典，根据key提取数据
sample = io.loadmat('sample.mat')['array_sample']
src_image = Image.open('fish.bmp')
RGB_img = np.array(src_image)
Gray_img = np.array(src_image.convert('L'))

# 通过mask，获取ROI区域
Gray_ROI = (Gray_img * mask)/256
RGB_mask = np.array([mask, mask, mask]).transpose(1, 2, 0)
RGB_ROI = (RGB_img * RGB_mask)/255

# 假设两类数据初始占比相同，即先验概率相同
P_pre1 = 0.5
P_pre2 = 0.5

# 假设每个数据来自两类的初始概率相同，即软标签相同
soft_guess1 = 0.5
soft_guess2 = 0.5

# 选择1维或是多维
gray_status = True
gray_status = False
#https://blog.csdn.net/sinat_35907936/article/details/109266603
