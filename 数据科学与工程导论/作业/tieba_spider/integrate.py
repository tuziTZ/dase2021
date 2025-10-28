import shutil
import os
cnt=0
path='./tables/'
tmp='./tables/tmp/'
dst='./tables/dst/'
dir_list=os.listdir(path)
for i in dir_list:
    path1=path+i
    if '原神内鬼' in path1:
        if 'txt' not in path1:
            file_list = os.listdir(path1)
            for j in file_list:
                shutil.move(path1+'/'+j,tmp)
                os.rename(tmp+j,tmp+str(cnt)+'.txt')

                shutil.move(tmp+str(cnt)+'.txt',dst)
                cnt += 1



