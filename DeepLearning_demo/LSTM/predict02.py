import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from torch import nn
import sys
 
# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x
if __name__ == '__main__':
    #1.准备数据
    data_len = 200
    t = np.linspace(0, 12*np.pi, data_len)#范围是0到12pi，中间取200个点
    sin_t = np.sin(t)
    cos_t = np.cos(t)
 
    dataset = np.zeros((data_len, 2))
    dataset[:,0] = sin_t 
    dataset[:,1] = cos_t 
    dataset = dataset.astype('float32')
 
    #choose dataset for training and testing 选择训练集和测试集
    train_data_ratio = 0.5 # 一半训练一半测试Choose 80% of the data for testing 
    train_data_len = int(data_len*train_data_ratio) #训练集长度
    train_x = dataset[:train_data_len, 0] #100*1
    train_y = dataset[:train_data_len, 1] #100*1

    #test_x = train_x  #test_y = train_y
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]
 
    #----------------- train -------------------
    train_x_tensor = train_x.reshape(-1, 5, 1) # set batch size to 5
    train_y_tensor = train_y.reshape(-1, 5, 1) # set batch size to 5
 
    #transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)#用于训练的数据 20*5*1
    train_y_tensor = torch.from_numpy(train_y_tensor)#用于训练的数据 20*5*1
 
    #2.定义模型
    lstm_model = LstmRNN(1, 16, output_size=1, num_layers=1) # 16 hidden units
    lstm_model.load_state_dict(torch.load("modelResult.pkl"))
    for x in lstm_model.parameters():
        print(x.shape,x.numel())

    #4.预测
    print(np.array(train_x_tensor.tolist()).shape)
    y = lstm_model(train_x_tensor)  #print(predictive_y_for_training.shape)
    print(np.array(y.tolist()).shape)

    s = t[:train_data_len]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(s, y.view(-1, 1).data.numpy(), 'g--')
    plt.plot(s, train_y_tensor.view(-1, 1).data.numpy(), 'b--')
    
    plt.show()

    