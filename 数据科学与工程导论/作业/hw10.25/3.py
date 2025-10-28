import matplotlib.pyplot as plt

Sepal_Length=[]
Sepal_Width=[]
Petal_Length=[]
Petal_Width=[]
Species=[]
color_list=[]
with open('iris.csv','r',encoding='utf-8') as fp:
    fp.readline()
    for i in range(2,152):
        s=fp.readline()
        list=s.split(',')
        Sepal_Length.append(list[0])
        Sepal_Width.append(list[1])
        Petal_Length.append(list[2])
        Petal_Width.append(list[3])
        Species.append(list[4])

for i in Species:
    if i=='setosa\n':
        color_list.append('b')
    elif i=='versicolor\n':
        color_list.append('g')
    else:
        color_list.append('r')


plt.subplot(231)
plt.scatter(Sepal_Length,Sepal_Width,color=color_list)
plt.xlabel='Sepal_Length'
plt.ylabel='Sepal_Width'

plt.subplot(232)
plt.scatter(Sepal_Length,Petal_Length,color=color_list)
plt.xlabel='Sepal_Length'
plt.ylabel='Petal_Length'

plt.subplot(233)
plt.scatter(Sepal_Length,Petal_Width,color=color_list)
plt.xlabel='Sepal_Length'
plt.ylabel='Petal_Width'

plt.subplot(234)
plt.scatter(Sepal_Width,Petal_Length,color=color_list)
plt.xlabel='Sepal_Width'
plt.ylabel='Petal_Length'

plt.subplot(235)
plt.scatter(Sepal_Width,Petal_Width,color=color_list)
plt.xlabel='Sepal_Width'
plt.ylabel='Petal_Width'

plt.subplot(236)
plt.scatter(Petal_Width,Petal_Length,color=color_list)
plt.xlabel='Petal_Width'
plt.ylabel='Petal_Length'

plt.show()

print('从图中可以看出，Petal_Width与Petal_Length两种属性区分Species的效果较好。')