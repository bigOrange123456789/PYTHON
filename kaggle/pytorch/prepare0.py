import pandas as pd
import numpy as np

data=pd.read_csv('../input/train.csv').drop(index=range(37000))
train_x=data.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
#.reshape((-1,28,28))   n 28 28
#.astype(np.uint8)      n 28 28
#[:,:,:,None]           n 28 28 1
train_y=data.iloc[:,0].values
np.savez('data',train_x=train_x, train_y=train_y)
