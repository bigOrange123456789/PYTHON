#https://www.kaggle.com/sehandev/ny-stock-price-prediction-gru-tensorflow-2-x
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# tensorflow 1
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

# tensorflow 2
from tensorflow import keras
from tensorflow.keras import layers

import sys

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10 #训练集 测试集 验证集
test_set_size_percentage = 10#训练集上训练模型，在验证集上评估模型，一旦找到的最佳的参数，就在测试集上最后测试一次

# import all stock prices
df = pd.read_csv("./prices-split-adjusted.csv", index_col = 0)
df.info()
'''
Data columns (total 6 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   symbol  851264 non-null  object 
 1   open    851264 non-null  float64
 2   close   851264 non-null  float64
 3   low     851264 non-null  float64
 4   high    851264 non-null  float64
 5   volume  851264 non-null  float64
'''

# number of different stocks
print('\nnumber of different stocks: ', len(list(set(df.symbol))))
print(list(set(df.symbol))[:10])

plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(df[df.symbol == 'EQIX'].open.values, color='red', label='open')
plt.plot(df[df.symbol == 'EQIX'].close.values, color='green', label='close')
plt.plot(df[df.symbol == 'EQIX'].low.values, color='blue', label='low')
plt.plot(df[df.symbol == 'EQIX'].high.values, color='black', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.plot(df[df.symbol == 'EQIX'].volume.values, color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best')

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
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)
'''
df_stock.columns.values =  ['open', 'close', 'low', 'high']
x_train.shape =     (1394, 19, 4)     # 通过前19天的数据预测明天的数据
y_train.shape =     (1394, 4)         # 1394+174+174=1742 约等于 1762
x_valid.shape =     (174, 19, 4)
y_valid.shape =     (174, 4)
x_test.shape =      (174, 19, 4)
y_test.shape =      (174, 4)
'''
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
tf1.reset_default_graph()

X = tf1.placeholder(tf1.float32, [None, n_steps, n_inputs])
y = tf1.placeholder(tf1.float32, [None, n_outputs])

# 实现方法1：
# use GRU cell
gru_layers = [
    tf1.nn.rnn_cell.GRUCell(num_units=n_neurons, activation=tf1.nn.leaky_relu)
    for layer in range(n_layers)
]
multi_layer_cell = tf1.nn.rnn_cell.MultiRNNCell(gru_layers)
rnn_outputs, states = tf1.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf1.float32)

stacked_rnn_outputs = tf1.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf1.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf1.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence

loss = tf1.reduce_mean(input_tensor=tf1.square(outputs - y)) # loss function = mean squared error
optimizer = tf1.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# run graph
with tf1.Session() as sess:
    sess.run(tf1.global_variables_initializer())
    for iteration in range(int(n_epochs * train_set_size / batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        if iteration % int(10 * train_set_size / batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
            print(f'{iteration * batch_size / train_set_size:.2f} epochs', end=': ')
            print(f'MSE train/valid = {mse_train:.6f}/{mse_valid:.6f}')

    # Tensorflow 1 model summary
    print('\n[ Tensorflow 1 model summary ]')
    print(rnn_outputs)
    print(stacked_rnn_outputs)
    print(outputs)

    # Tensorflow 1 prediction result
    y_train_pred_tf1 = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred_tf1 = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred_tf1 = sess.run(outputs, feed_dict={X: x_test})

    print('\n[ Tensorflow 1 prediction result ]')
    print(y_train_pred_tf1.shape)
    print(y_valid_pred_tf1.shape)
    print(y_test_pred_tf1.shape)

# 实现方法2：
# Build the GRU model - tensorflow 2
def build_model():
    model = keras.Sequential()

    # GRU Layers
    model.add(layers.GRU(n_neurons, activation='relu', input_shape=(n_steps, n_inputs), return_sequences=True))
    for i in range(n_layers - 1):
        model.add(layers.GRU(n_neurons, activation='relu', input_shape=(n_steps, n_inputs), return_sequences=False))

    # Output Layer
    model.add(layers.Dense(n_outputs))
    print("model.summary()")
    model.summary()

    return model

model = build_model()

model.compile(
    loss="mean_squared_error",
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
# Tensorflow 2 model summary
print('[ Tensorflow 2 model summary ]')
print(model.summary())

# Tensorflow 2 prediction result
# y_train_pred_tf2 = model.predict_on_batch(x_train)
# y_valid_pred_tf2 = model.predict_on_batch(x_valid)
# y_test_pred_tf2 = model.predict_on_batch(x_test)
y_train_pred_tf2 = model.predict(x_train)
y_valid_pred_tf2 = model.predict(x_valid)
y_test_pred_tf2 = model.predict(x_test)

print('[ Tensorflow 2 prediction result ]')
print(y_train_pred_tf2.shape)
print(y_test_pred_tf2.shape)
print(y_valid_pred_tf2.shape)

ft = 0 # 0 = open, 1 = close, 2 = highest, 3 = lowest

## show predictions
plt.figure(figsize=(15, 5))

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:, ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred_tf2.shape[0], y_train_pred_tf2.shape[0]+y_test_pred_tf2.shape[0]),
         y_test_pred_tf2[:, ft], color='red', label='tf2 test prediction')

plt.plot(np.arange(y_train_pred_tf1.shape[0], y_train_pred_tf1.shape[0]+y_test_pred_tf1.shape[0]),
         y_test_pred_tf1[:,ft], color='green', label='tf1 test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')

plt.show()

