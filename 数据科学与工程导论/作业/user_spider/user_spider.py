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


class UserSpider():

    def __init__(self, path):
        self.driver = webdriver.Firefox()
        self.last_text = ''
        self.txt_list = os.listdir(path)
        self.path = path
        self.user_file_path = ''
        self.url = 'https://tieba.baidu.com'
        self.visited_list = []
        with open('user_tables/visited_user.txt', 'r', encoding='utf-8') as f:
            r = f.read()
        r = r.split('\n')
        for i in r:
            self.visited_list.append(i)
        pass

    def visit_page(self):
        self.yanzheng()
        time.sleep(2)
        if 'page404' in self.driver.page_source:
            return
        page_text = self.driver.page_source
        self.last_text = page_text
        tree = etree.HTML(page_text)
        try:
            sex1 = tree.xpath('//div[@class="userinfo_userdata"]/span[contains(@class,"sex")]/@class')[0]

            if 'female' not in sex1:
                sex = '男'
            else:
                sex = '女'
        except IndexError:
            sex = ''
        span_list = tree.xpath('//div[@class="userinfo_userdata"]/span/text()')
        with open(self.user_file_path, 'w', encoding='utf-8') as fp:
            fp.write(sex + ',')
            for i in span_list:
                fp.write(i + ',')
            fp.write('\n')

        self.get_follow_content()
        self.get_ba_content()

    def get_ba_content(self):
        try:
            page_text = self.driver.page_source
            self.last_text = page_text
            tree = etree.HTML(page_text)
            ba_list = tree.xpath('//div[@id="forum_group_wrap"]/a//text()')
            print(ba_list)
            with open(self.user_file_path, 'a', encoding='utf-8') as fp:
                for i in ba_list:
                    ba_name = i
                    fp.write(ba_name + ',')
                fp.write('\n')
        except IndexError:
            return

    def get_follow_content(self):
        try:
            page_text = self.driver.page_source
            self.last_text = page_text
            tree = etree.HTML(page_text)
            user_list = tree.xpath('//li[@class="concern_item"]/a/@href')
            with open(self.user_file_path, 'a', encoding='utf-8') as fp:
                for i in user_list:
                    user1 = re.findall(r'.*?id=(.*?)&.*?', i)[0]
                    user = user1.split('.')[2]
                    fp.write(user + ',')
                fp.write('\n')
        except IndexError:
            return

    def yanzheng(self):

        if '百度安全验证' in self.driver.page_source:
            a = my_web()
            a.main(self.driver)

    def run(self):
        try:
            self.driver.get(self.url)
            self.yanzheng()
            time.sleep(5)
            # 扫码登录
            button = self.driver.find_element(By.XPATH, '//div[@class="u_menu_item"]/a[@href="#"]')
            button.click()
            time.sleep(15)
            self.yanzheng()
            for i in self.txt_list:
                target_file = self.path + i
                with open(target_file, 'r', encoding='utf-8') as fp:
                    r = fp.read()
                    r_list = r.split('\n')[2:-1]
                for line in r_list:
                    s = line.split(',')
                    target_join = s[1]
                    if '/home/main' not in target_join:
                        continue
                    # 确认该用户是否被访问过
                    user_id1 = re.findall(r'.*?id=(.*?)&fr=.*?', target_join)[0]
                    user_id = user_id1.split('.')[2]
                    if user_id in self.visited_list:
                        continue
                    self.visited_list.append(user_id)
                    with open('user_tables/visited_user.txt', 'a', encoding='utf-8') as f:
                        f.write(user_id + '\n')

                    # 写入用户信息的txt文件
                    self.user_file_path = 'user_tables/' + user_id + '.txt'

                    target_url = self.url + target_join
                    original_window = self.driver.current_window_handle
                    # 在新的标签页打开链接
                    self.driver.execute_script(f'window.open("{target_url}", "_blank");')
                    # 切换到新的标签页
                    self.driver.switch_to.window(self.driver.window_handles[-1])
                    # 爬取新的标签页
                    print(target_url)
                    self.visit_page()
                    # 关闭当前标签页
                    self.driver.close()
                    # 切回到之前的标签页
                    self.driver.switch_to.window(original_window)
            # self.driver.quit()

        except Exception as e:
            with open('test.html', 'w', encoding='utf-8') as ep:
                ep.write(self.last_text)

            traceback.print_exc()


if __name__ == '__main__':
    spider = UserSpider("sources/bilibili/")
    spider.run()
