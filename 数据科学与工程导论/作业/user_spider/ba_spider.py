#!/usr/bin/python3
# -*- coding: utf-8 -*-

import traceback
import requests
from lxml import etree
import json
import re
import os
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from id import my_web


class BaSpider():

    def __init__(self, kw):
        self.kw = kw
        self.page_url = "https://tieba.baidu.com/f?kw={}&ie=utf-8"
        self.path='user_tables/'+kw
        self.pn=0
        pass

    def get_ba_content(self, driver,ba_name):
        if '百度安全验证' in driver.page_source:
            a = my_web()
            a.main(driver)
        time.sleep(2)
        text = driver.page_source
        if '抱歉，根据相关法律法规和政策，相关结果不予展现。'in text:
            return
        tree = etree.HTML(text)

        with open('ba_info/'+self.kw+'.txt', 'a', encoding='utf-8') as bafp:
            bafp.write(ba_name + ' ')
            num = tree.xpath('//div[@class="th_footer_l"]/span/text()')
            for i in num:
                bafp.write(i + ' ')
            index=tree.xpath('//div[@class="card_info"]/ul/li/span/text()')
            try:
                bafp.write(index[1])
            except:
                bafp.write('其他')
            bafp.write('\n')

    def run(self):
        driver=webdriver.Firefox()
        driver.get('https://tieba.baidu.com')
        time.sleep(2)

        if '百度安全验证' in driver.page_source:
            a = my_web()
            a.main(driver)

        visited_ba=[]
        file_list=os.listdir(self.path)
        for i in file_list:
            if 'visit' in i:
                continue
            with open(self.path+ '/' + i, 'r', encoding='utf-8') as fp:
                r = fp.read()
            r_list = r.split('\n')
            if len(r_list) < 3:
                continue
            ba_list=r_list[2].split(',')[:-1]
            for j in ba_list:
                if j not in visited_ba:
                    visited_ba.append(j)
        for i in visited_ba:
            driver.get(self.page_url.format(i))
            self.get_ba_content(driver,i)
            print(i)


if __name__ == '__main__':
    spider = BaSpider("剑网3")
    spider.run()