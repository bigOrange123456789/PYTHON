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
def f(x):
    return 2*(x**2)%10
if __name__ == '__main__':
    #1.准备数据
    x=[]
    y=[]
    for i in range(500):
        x.append(f(i))
        y.append(f(i+100))
    x=np.array(x)
    y=np.array(y)
    dataset = np.zeros((x.shape[0], 2))
    dataset[:,0] = x 
    dataset[:,1] = y 
    dataset = dataset.astype('float32')
    train_x = dataset[:, 0] 
    train_y = dataset[:, 1] 
    train_x_tensor = train_x.reshape(-1, 10, 1) # set batch size to 5
    train_y_tensor = train_y.reshape(-1, 10, 1) # set batch size to 5
    train_x_tensor = torch.from_numpy(train_x_tensor)#用于训练的数据 20*5*1
    train_y_tensor = torch.from_numpy(train_y_tensor)#用于训练的数据 20*5*1

 
    #2.定义模型
    lstm_model = LstmRNN(1, 32, output_size=1, num_layers=1) # 16 hidden units
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
 
    #3.训练
    for epoch in range(1000):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if loss.item() < 1e-4:
            print('Epoch [{}], Loss: {:.5f}'.format(epoch+1, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % 100 == 0:
            print('Epoch: [{}], Loss:{:.5f}'.format(epoch+1, loss.item()))
    
    #4.预测
    y2 = lstm_model(train_x_tensor)  #print(predictive_y_for_training.shape)
 
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(500), x , 'm--', label='pre_cos_tst')  
    plt.plot(range(500), y2.view(-1, 1).data.numpy(), 'r--', label='pre_cos_tst')  
    plt.show()