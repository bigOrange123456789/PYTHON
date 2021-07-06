import numpy as np
covMat = np.cov([[1,1],[0,0],[-1,-1]], rowvar=0) 
eigVals,eigVects = np.linalg.eig(covMat) #注意输出的为列向量

print(eigVals)
print(eigVects)