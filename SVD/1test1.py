import numpy as np

A=np.random.randint(1,25,[5,5])
u,s,vt = np.linalg.svd(A)

s2 = np.zeros([5,5])
for i in range(5):
    s2[i][i] = s[i]
tmp = np.dot(u,s2)
tmp = np.dot(tmp,vt)

print(A)
print(tmp)
