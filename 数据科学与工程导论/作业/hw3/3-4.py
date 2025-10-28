import random
import math
def a(x):
    return pow(x,2)+4*x*math.sin(x)
N=1000000
hi=30
count=0
for i in range(1,N):
    x=random.uniform(2.0,3.0)
    y=random.uniform(0.0,hi)
    if a(x)>y:
        count=count+1
print((count/N)*hi)