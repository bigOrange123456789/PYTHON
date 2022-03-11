import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D,GRU, MaxPool2D
import keras
#from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


#https://www.kaggle.com/sehandev/ny-stock-price-prediction-gru-tensorflow-2-x
print("只保留第二种预测方法")
import sys

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10 #训练集 测试集 验证集
test_set_size_percentage = 10#训练集上训练模型，在验证集上评估模型，一旦找到的最佳的参数，就在测试集上最后测试一次

# import all stock prices
df = pd.read_csv("./prices-split-adjusted.csv", index_col = 0)
df.info()

# function to create train, validation, test data given stock data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.values # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data)
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]

    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]

    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# choose one stock
df_stock = df[df.symbol == 'EQIX'].copy()
df_stock.drop(['symbol'], 1, inplace=True)
df_stock.drop(['volume'], 1, inplace=True)
#只保留4个指标：日期和高、中、低股值 #(1762, 4)
cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)
print(df_stock)
print(df_stock.shape)

# create train, test data
seq_len = 20 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock, seq_len)

# parameters
n_steps = seq_len - 1
n_inputs = 4
n_neurons = 200
n_outputs = 4
n_layers = 2
learning_rate = 1e-3
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

index_in_epoch = 0
perm_array = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

# Build the GRU model - tensorflow 1

# 实现方法1：
# Build the GRU model - tensorflow 2
def build_model():
    model = Sequential()
    # GRU Layers
    model.add(GRU(n_neurons, activation='relu', input_shape=(n_steps, n_inputs), return_sequences=True))#input_shape=(19,4)   n_neurons=200? 
    #for i in range(n_layers - 1):   
    #    model.add(layers.GRU(n_neurons, activation='relu', input_shape=(n_steps, n_inputs), return_sequences=False))
    # Output Layer
    model.add(Dense(n_outputs))#n_outputs 4
    return model

model = build_model()
print(model)
print("type:",type(model))
#for i in dir(model):
#    print(i)
w=model.get_weights()
print(len(w))
for i in w:
    log=str(len(i))
    #if type(i[0])!=type(0):
    #    log=log+"*"+str(len(i[0]))
        #if type(i[0][0])!=type(0):
            #print(type(i[0][0]))
            #log=log+"*"+str(len(i[0][0]))
    print("\t",log)
#print("weighted_metrics",model.weighted_metrics)
#sys.exit(0)
# Define the optimizer
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(
    loss="mean_squared_error",
    #optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=['accuracy'],
)

model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    batch_size=batch_size,
    epochs=n_epochs,
    verbose=0,
)

y_test_pred_tf2 = model.predict(x_test) # 174*19*4 -> 174*4

ft = 1 # 0 = open, 1 = close, 2 = highest, 3 = lowest

print("x_train")
#print(x_train)
print(x_train.shape)

print("x_test")#
#print(x_test)
print(x_test.shape)

print("y_test_pred_tf2")#174*4
print(y_test_pred_tf2.shape)

## show predictions
plt.figure(figsize=(15, 5))


#plt.subplot(1,2,1)

plt.subplot(2,1,2)
plt.plot(range(len(y_test[:, ft])),
         y_test[:, ft], color='black', label='test target')

plt.plot(range(len(y_test[:, ft])),
         y_test_pred_tf2[:, ft], color='red', label='tf2 test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')

plt.show()

