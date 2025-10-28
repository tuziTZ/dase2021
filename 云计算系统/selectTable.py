#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import clickhouse_driver

# 建立连接
conn = clickhouse_driver.connect(database="ck_test", user="admin", password="123456ABcd", host="10.23.76.137", port="9000")
cur = conn.cursor()
cur.execute("SELECT ID, NAME, ADDRESS, SALARY  from COMPANY order by ID")
rows = cur.fetchall()
print ("ID      NAME      ADDRESS      SALARY")
for row in rows:
    print (row[0], "\t", row[1], "\t", row[2], "\t", row[3])
conn.close()
