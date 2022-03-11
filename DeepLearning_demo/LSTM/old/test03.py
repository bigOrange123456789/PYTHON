import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import sys
 
# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):#- input_size: feature size  -hidden_size: number of hidden units  -output_size: number of output  -num_layers: layers of LSTM to stack
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        #print(s,b,h)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x
 
if __name__ == '__main__':
    #create database
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
 
    lstm_model = LstmRNN(1, 16, output_size=1, num_layers=1) # 16 hidden units
    #lstm_model = LstmRNN(2, 3, output_size=1, num_layers=1) 
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    #print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in lstm_model.parameters())))
    #for x in lstm_model.parameters():
    #    print(x.shape,x.numel())
    #for x in lstm_model.parameters():
    #    print(x.shape,x.numel())
    for name, parameters in lstm_model.named_parameters():
        #print(name)
        #print(parameters)
        print(name,"\t\t",parameters.shape,"\t\t",parameters.numel())

    '''
    import math
    import random
    for i in range(10):
        n=2+i
        h=3+10*i
        m=1+3*i
        lstm_model = LstmRNN(n, h, output_size=m, num_layers=1) 
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in lstm_model.parameters())))
        for x in lstm_model.parameters():
            print(x.shape)
            #print(i,[n,h,m],x.shape,x.numel())
            #break
        print()
    sys.exit(0)
    '''
 
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
 
    max_epochs = 10000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        #sys.exit(0)
        loss = loss_function(output, train_y_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
    
    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, 1).data.numpy()
    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files
    # ----------------- test -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval() # switch to testing model
 
    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 5, 1) # set batch size to 5, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor)
 
    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, 1).data.numpy()
 
    # ----------------- plot -------------------
    t_for_training = t[:train_data_len]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t_for_training, train_x, 'g', label='sin_trn')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_trn')
 
    plt.plot(t_for_testing, test_x, 'c', label='sin_tst')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_tst')
 
    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line') # separation line
 
    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size = 15, alpha = 1.0)
    plt.text(20, 2, "test", size = 15, alpha = 1.0)
  
    plt.show()