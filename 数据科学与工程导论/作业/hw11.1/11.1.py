import pymysql
db = pymysql.connect(host="cdb-r2g8flnu.bj.tencentcdb.com",port=10209,user="dase2020",password="dase2020",database="dase_intro_2020")
cursor = db.cursor()
sql = "SELECT city,hour,is_workday,temp_air,temp_body,weather,wind,y FROM bicycle_train LIMIT 17,5"
cursor.execute(sql)  # 执行SQL语句
result = cursor.fetchall()
for row in result:
    city= row[0]
    hour= row[1]
    is_workday= row[2]
    temp_air= row[3]
    temp_body= row[4]
    weather= row[5]
    wind= row[6]
    y = row[7]
    print("city=%d,hour=%d,is_workday=%d,temp_air=%d,temp_body=%d,weather=%d,wind=%d,y=%d"%( city,hour,is_workday,temp_air,temp_body,weather,wind,y))

sql="SELECT MIN(wind),MAX(wind) FROM bicycle_train"
cursor.execute(sql)
result=cursor.fetchall()
s=result[0]
print("wind的取值范围是[%d,%d]"%(s[0],s[1]))

sql="SELECT AVG(temp_air) FROM bicycle_train WHERE city=0 AND hour=10 AND weather=1 AND y>=100 AND wind<=1"
cursor.execute(sql)
result=cursor.fetchall()
s=result[0]
print("在此条件下大气温度的平均值是%.1f度"%s[0])

sql="SELECT temp_air FROM bicycle_train WHERE city=0 AND hour=10 AND weather=1 AND y>=100 AND wind<=1"
cursor.execute(sql)
result=cursor.fetchall()
sum=0
cnt=0
for i in result:
    sum+=pow((i[0]-s[0]),2)
    cnt+=1
a=sum/cnt
print("在此条件下大气温度的方差是%.2f度"%a)

sql0="SELECT SUM(y) FROM bicycle_train WHERE city=0 AND weather=3 AND is_workday=1"
cursor.execute(sql0)
result0=cursor.fetchall()

sql1="SELECT SUM(y) FROM bicycle_train WHERE city=1 AND weather=3 AND is_workday=1"
cursor.execute(sql1)
result1=cursor.fetchall()
if result1[0][0]<result0[0][0]:
    print("北京为%d,上海为%d"%(result0[0][0],result1[0][0]))
else:print("上海为%d,北京为%d"%(result1[0][0],result0[0][0]))

list=[]
sql1="SELECT AVG(y) FROM bicycle_train WHERE city=1 AND is_workday=1 AND temp_body<=10 AND hour=17"
cursor.execute(sql1)
result=cursor.fetchall()
s1=result[0][0]
sql2="SELECT AVG(y) FROM bicycle_train WHERE city=1 AND is_workday=1 AND temp_body<=10 AND hour=18"
cursor.execute(sql2)
result=cursor.fetchall()
s2=result[0][0]
sql3="SELECT AVG(y) FROM bicycle_train WHERE city=1 AND is_workday=1 AND temp_body<=10 AND hour=19"
cursor.execute(sql3)
result=cursor.fetchall()
s3=result[0][0]
print("17时%.0f辆,18时%.0f辆,19时%.0f辆"%(s1,s2,s3))