import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,10)
sum_list=[]
aver_list=[]
with open('daily_KP_SUN_2020.csv','r',encoding='utf-8') as fp:
    for i in range(3):
        fp.readline()
    month=0
    day=0
    sum=0
    for i in range(4,278):
        s=fp.readline()
        read_list=s.split(',')
        if month!=int(read_list[1]):
            month+=1
            if day!=0:

                aver_list.append(sum/day)
                sum_list.append(sum)
            sum=0
            day=0
        else:
            sum+=float(read_list[3])
            day+=1
    aver_list.append(sum / day)
    sum_list.append(sum)
y=np.array(sum_list)
plt.subplot(211)
plt.ylabel('Total light time per month')
plt.bar(x,y)
plt.subplot(212)
plt.xlabel('month')
plt.ylabel('Average light time per month')

plt.bar(x,np.array(aver_list))
plt.show()

