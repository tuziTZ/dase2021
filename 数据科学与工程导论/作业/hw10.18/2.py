import time
with open('stuGrade.csv', 'r',encoding='utf-8') as fp:
    fp.readline()
    chinese = 0
    math = 0
    english = 0

    for i in range(0, 5):
        x = fp.readline()
        x_list = x.split(',')
        chinese+=int(x_list[1])
        math+=int(x_list[2])
        english+=int(x_list[3])
    chinese/=5
    math/=5
    english/=5

print("语文学科平均成绩是：" + '%.2f'%chinese)
print("数学学科平均成绩是：" + '%.2f'%math)
print("英语学科平均成绩是：" + '%.2f'%english)


with open('my.txt', 'w',encoding='utf-8') as fp1:
    s='{:.2f}'.format(chinese)+' '+'{:.2f}'.format(math)+' '+'{:.2f}'.format(english)+'\n'

    fp1.write('10215501412彭一珅\n')
    fp1.write(s)
    t4 = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    fp1.write(t4+'\n')
    time.sleep(2)
    t5=time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    fp1.write(t5+'\n')

