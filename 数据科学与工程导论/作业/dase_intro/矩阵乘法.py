import numpy as np
x=np.array([1/4,1/4,1/4,1/4])
A = np.array([[0, 1/3, 1/3,1/3], [1/2, 0, 1/2,0],[0,0,0,1],[1/2,1/2,0,0]])
B = np.array([[0, 1/3, 1/3,1/3], [1/2, 0, 1/2,0],[0,0,0,1],[1/2,1/2,0,0]])
for i in range(0,10):
    C = np.dot(B, A)
    B=C
print( np.dot(x,A))
print( np.dot(x,B))
