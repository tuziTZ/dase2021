import pymysql,csv
db = pymysql.connect(host="cdb-r2g8flnu.bj.tencentcdb.com",port=10209,user="dase2020",password="dase2020",database="dase_intro_2020")
cursor = db.cursor(cursor=pymysql.cursors.DictCursor)
sql="SELECT * FROM SH_Grade"
cursor.execute(sql)
result=cursor.fetchall()
#题目1
with open("SH_Grade.csv",'w') as fp:
    writer = csv.writer(fp)
    classlist=[]
    for x in result[0]:
        classlist.append(x)
    classlist[0]=classlist[1]
    classlist[1]="Class"
    writer.writerow(classlist)
    for i in result:
        list1=[]
        for x in i.values():
            list1.append(x)
        list1[0]=list1[1]
        list1[1]=list1[1][0]
        writer.writerow(list1)
#题目2
import pandas as pd
studata=pd.read_csv('SH_Grade.csv')
num1=studata.shape[0]
studata1=studata.drop_duplicates('StuId')
num2=studata1.shape[0]
print("处理前%d条，处理后%d条"%(num1,num2))

#题目3
num1=studata1.shape[0]
studata2=studata1.dropna(thresh=46)
num2=studata2.shape[0]
print("处理前%d条，处理后%d条"%(num1,num2))
#题目4
studata3=studata2.copy()
studata3['Sex']=studata2['Sex'].fillna(method='ffill')
studata4=studata3.copy()
for i in range(3,len(classlist)):
    studata4[classlist[i]] = studata3[classlist[i]].fillna(studata3[classlist[i]].median())

#题目5
studata5=studata4.copy()
for i in range(27,len(classlist)):
    c=classlist[i]
    if c[0]=='P':
        if studata5[c].max()<90:
            studata5[c]=studata5[c]/90*100
    elif c[2]=='E':
        if studata5[c].max()<60:
            studata5[c] = studata5[c] / 60 * 100
    elif (i>=39)&(i<=41):
        studata5[c] = studata5[c] / 120 * 100
    else:
        if studata5[c].max()>100:
            studata5[c] = studata5[c] / 150 * 100

#题目6
import matplotlib.pyplot as plt
class_female=studata5.loc[studata5['Sex']=='F'].groupby('Class')['Sex'].count()
class_male=studata5.loc[studata5['Sex']=='M'].groupby('Class')['Sex'].count()

plt.bar(class_female.index,class_female.values,tick_label=['A','B','C','D','E','F','G'],label='Female')
plt.bar(class_male.index,class_male.values,tick_label=['A','B','C','D','E','F','G'],bottom=class_female.values,label='Male')
plt.legend()
plt.show()

plt.legend()
#题目7
A13_list=[]
A15_list=[]
name_list=[]
for i in classlist:
    if i[:3]=='CHI':
        A13_list.append(studata5.loc[studata5['StuId']=='A13'][i])
        name_list.append(i)
        A15_list.append(studata5.loc[studata5['StuId']=='A15'][i])
plt.plot(name_list,A13_list)
plt.plot(name_list,A15_list)
plt.show()
#题目8
StuId_list=[]
Class_list=[]
Eng_list=[]
Chi_list=[]
data1=studata5.loc[(studata5['ENG721']<60)|(studata5['CHI721']<60)]
for i in data1['StuId']:
    StuId_list.append(i)
for i in data1['Class']:
    Class_list.append(i)
for i in data1['ENG721']:
    Eng_list.append(i)
for i in data1['CHI721']:
    Chi_list.append(i)
for i in range(len(StuId_list)):
    print(StuId_list[i],Class_list[i],Eng_list[i],Chi_list[i])

#题目9
A=studata5.loc[studata5['Class']=='A']
C=studata5.loc[studata5['Class']=='C']
CHI_list=[A['CHI622'].mean(),A['CHI622'].var(),C['CHI622'].mean(),C['CHI622'].var()]
MATH_list=[A['MATH622'].mean(),A['MATH622'].var(),C['MATH622'].mean(),C['MATH622'].var()]
ENG_list=[A['ENG622'].mean(),A['ENG622'].var(),C['ENG622'].mean(),C['ENG622'].var()]
print(CHI_list)
print(MATH_list)
print(ENG_list)
#C班语文和数学平均分比A班低，但是英语平均分比A班高
#C班数学成绩方差大，学生水平差距大

#题目10
with open("task8.csv",'w') as fp:
    writer = csv.writer(fp)
    classl=['StuId','Class','ENG721','CHI721']
    writer.writerow(classl)

    for i in range(len(StuId_list)):
        list2 = []
        list2.append(StuId_list[i])
        list2.append(Class_list[i])
        list2.append(Eng_list[i])
        list2.append(Chi_list[i])
        writer.writerow(list2)

