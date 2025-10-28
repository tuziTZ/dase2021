#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import clickhouse_driver

# 建立连接
conn = clickhouse_driver.connect(database="ck_test", user="admin", password="123456ABcd", host="10.23.76.137", port="9000")

# 创建游标
cur = conn.cursor()

# 创建表
cur.execute('CREATE TABLE COMPANY ON CLUSTER ck_cluster \
																	(ID Int32 NOT NULL,\
                                   NAME String NOT NULL,\
                                   AGE Int16 NOT NULL,\
                                   ADDRESS String,\
                                   SALARY Float64, \
                                   PRIMARY KEY (ID))  \
           												 ENGINE = MergeTree()    \
																	 ORDER BY (ID);')
# 提交事务
conn.commit()

# 关闭连接
conn.close()
