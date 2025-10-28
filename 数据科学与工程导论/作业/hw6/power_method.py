import numpy as np
A=np.matrix([[5,-1,1],[-1,2,0],[1,0,3]])
x=np.matrix([[1],[1],[1]])
k=0
while(k<30):
    y=A*x
    x=y/y.max()
    r1=y.max()
    k+=1

k=0
A_1=np.linalg.inv(A)
while(k<100):
    y=A_1*x
    x=y/y.max()
    r2=y.max()
    k+=1

r2=1/r2
print(r1/r2)