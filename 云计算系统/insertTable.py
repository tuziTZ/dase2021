#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import clickhouse_driver

# 建立连接
conn = clickhouse_driver.connect(database="ck_test", user="admin", password="123456ABcd", host="10.23.76.137", port="9000")

cur = conn.cursor()
cur.execute("INSERT INTO COMPANY VALUES (1, 'Paul', 32, 'California', 20000.00 )");
cur.execute("INSERT INTO COMPANY VALUES (2, 'Allen', 25, 'Texas', 15000.00 )");
cur.execute("INSERT INTO COMPANY VALUES (3, 'Eric', 35, 'Florida', 25000.00 )");
conn.commit()
conn.close()
