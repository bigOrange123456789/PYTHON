import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

# Load the data
train = pd.read_csv("../input/train.csv")
print("原始数据：",train.shape)
train=train.drop(index=range(37000))
print("提取数据：",train.shape)

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)# Drop 'label' column

X_train = (X_train/255.).values.reshape(-1,28,28,1)#Normalize& Reshape 3 dimensions (h= 28, w= 28 , canal = 1)
Y_train = to_categorical(Y_train, num_classes = 10)# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

#将数据差分为训练集和验证集 Split the train and the validation set for the fitting #验证validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=2)
for i in [X_train, X_val, Y_train, Y_val]:
    print(i.shape)
#test_size是测试数据在总数据中的比例 random_state是随机数种子
#X_train.shape 训练集输入, X_val.shape 测试集输入, Y_train.shape训练集输出, Y_val.shape 测试集输出
np.savez('data',X_train=X_train, Y_train=Y_train,X_val=X_val,Y_val=Y_val)
