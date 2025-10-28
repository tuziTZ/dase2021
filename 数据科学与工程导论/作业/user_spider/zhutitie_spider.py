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


class ZhutitieSpider():

    def __init__(self, kw):
        self.kw = kw
        self.page_url = "https://tieba.baidu.com/f?kw={}&ie=utf-8&pn={}"
        self.path='./Zhutitietables/'
        self.pn=0
        pass

    def get_ba_content(self, driver):

        if '百度安全验证' in driver.page_source:
            a = my_web()
            a.main(driver)
        time.sleep(2)
        text = driver.page_source
        tree = etree.HTML(text)
        name_list = tree.xpath('//li[@class=" j_thread_list clearfix thread_item_box"]//div[@class="threadlist_lz clearfix"]//a[@class="j_th_tit "]/@title')

        date_list=re.findall('title="创建时间">(.*?)</span>',text)
        print(len(name_list),len(date_list))
        with open(self.path+self.kw+'.txt', 'a', encoding='utf-8') as bafp:
            for i in range(0,min(len(date_list),len(name_list))):
                bafp.write(name_list[i]+','+date_list[i]+'\n')
        self.pn += 50

    def login(self,driver):
        # 扫码登录

        if '百度安全验证' in driver.page_source:
            a = my_web()
            a.main(driver)
        time.sleep(2)
        button = driver.find_element(By.XPATH, '//div[@class="u_menu_item"]/a[@href="#"]')
        button.click()
        time.sleep(15)

    def run(self):


        status = 0
        driver = webdriver.Firefox()
        driver.get(self.page_url.format(self.kw, self.pn))
        self.login(driver)
        while(self.pn<700000):

        # driver.get(self.ba_url.format(self.kw))
            time.sleep(1)
            if '百度安全验证' in driver.page_source:
                try:
                    a = my_web()
                    a.main(driver)
                except IndexError:
                    driver.refresh()

            self.get_ba_content(driver)
            driver.get(self.page_url.format(self.kw, self.pn))

        driver.quit()

            # traceback.print_exc()


if __name__ == '__main__':
    spider = ZhutitieSpider("bilibili")
    spider.run()