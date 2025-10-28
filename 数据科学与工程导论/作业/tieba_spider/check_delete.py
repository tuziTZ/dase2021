import shutil
import os
path='./tables/原神内鬼/'
dir_list=os.listdir(path)
title_list=[]
for i in range(0,len(dir_list)):
    path1=path+str(i)+'.txt'
    with open(path1,'r',encoding='utf-8') as fp:
        title_list.append(fp.readline())
list2=[]

reindex_list=[]
for i in range(0,len(title_list)):
    if title_list[i] not in list2:
        list2.append(title_list[i])
    else:
        reindex_list.append(i)
for i in reindex_list:
    os.remove(path+str(i)+'.txt')
