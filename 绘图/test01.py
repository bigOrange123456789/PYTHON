a=0
b=100
n=100
def f(x):
    #print(x)
    if x==0:
        return 0
    return 1./(1+1/(x**2))

d=(b-a)/n
x=[]
y=[]

for i in range(n):
    x0=a+i*d
    x.append(x0)
    y.append(f(x0))

import matplotlib.pyplot as plt
#print(x,y)
plt.plot(x, y)

plt.xlabel("X")
plt.ylabel("Y")
plt.title('T')
plt.show()