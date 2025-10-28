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


class TiebaSpider():

    def __init__(self, kw):
        self.kw = kw
        self.page_url = "https://tieba.baidu.com/f?kw={}&ie=utf-8&pn={}"
        self.ba_url = "https://tieba.baidu.com/f?kw={}&ie=utf-8"
        file_list = os.listdir('./tables/')
        self.max_dir = 0
        self.last_text = ''
        self.cnt = 0
        for i in file_list:
            if re.match(r"\d", i) != None:
                if self.max_dir < (int)(i):
                    self.max_dir = (int)(i)

        self.max_dir += 1
        self.max_pn = 0
        pass

    def get_ba_content(self, driver):

        # 扫码登录
        time.sleep(2)
        button = driver.find_element(By.XPATH, '//div[@class="u_menu_item"]/a[@href="#"]')
        button.click()
        time.sleep(15)
        if '百度安全验证' in driver.page_source:
            a = my_web()
            a.main(driver)
        time.sleep(2)
        text = driver.page_source
        self.last_text = text

        tree = etree.HTML(text)

        with open('./tables/%s.txt' % self.kw, 'w', encoding='utf-8') as bafp:
            num = tree.xpath('//div[@class="th_footer_l"]/span/text()')
            for i in num:
                bafp.write(i + ' ')
            bafp.write(str(self.max_dir) + '\n')
            self.max_pn = (int)(num[0])

        self.max_pn += 50

    def get_tie_content(self, driver):
        try:
            self.cnt += 1
            flag = 0
            time.sleep(2)

            if '百度安全验证' in driver.page_source:
                a = my_web()
                a.main(driver)
            time.sleep(1)
            page_text = driver.page_source
            self.last_text = driver.page_source
            tree = etree.HTML(page_text)

            if 'page404' in driver.page_source:
                return

            title = tree.xpath('//h3[contains(@class,"core_title_txt pull-left text-overflow  ")]/@title')[0]

            path = './tables/%s' % str(self.max_dir)
            if os.path.exists(path) == 0:
                os.mkdir(path)
            new_path = path + '/' + str(self.cnt) + '.txt'
            with open(new_path, 'w', encoding='utf-8') as fp:
                fp.write(title + '\n')
                fp.write('id,user_href,user_level,is_lz,user_name,content,date,floor,ip_address\n')
            num = 1
            while (1):
                time.sleep(1)
                if '百度安全验证' in driver.page_source:
                    a = my_web()
                    a.main(driver)
                time.sleep(1)
                # # 1、准备js代码
                # js_down = "window.scrollTo(0, 1000)"
                # # 2、执行js代码
                # driver.execute_script(js_down)
                # time.sleep(2)
                # # 1、准备js代码
                # js_down = "window.scrollTo(0, 1000)"
                # # 2、执行js代码
                # driver.execute_script(js_down)
                # time.sleep(2)
                text = driver.page_source
                self.last_text = text

                tree = etree.HTML(text)
                floor_list = tree.xpath('//div[@class="l_post l_post_bright j_l_post clearfix  "]')

                item_dict = {
                    'user_href': '',
                    'user_level': '',
                    'is_lz': '',
                    'user_name': '',
                    'content': '',
                    'date': '',
                    'floor': '',
                    'ip_address': ''
                }
                for floor in floor_list:
                    need = floor.xpath('div[@class="d_author"]/ul[@class="p_author"]/li[@class="d_name"]/a/@href')
                    if len(need) != 0:
                        item_dict["user_href"] = need[0]
                    else:
                        item_dict["user_href"] = ''

                    need = floor.xpath('div[@class="d_author"]/ul[@class="p_author"]//div[@class="d_badge_lv"]/text()')
                    if len(need) != 0:
                        item_dict["user_level"] = need[0]
                    else:
                        item_dict["user_level"] = ''

                    item_dict['is_lz'] = str(
                        len(floor.xpath('div[@class="d_author"]/div[@class="louzhubiaoshi_wrap"]')))

                    need = floor.xpath('div[@class="d_author"]/ul[@class="p_author"]/li[@class="d_name"]/a/text()')
                    if len(need) != 0:
                        item_dict['user_name'] = need[0]
                    else:
                        item_dict['user_name'] = ''

                    need = floor.xpath(
                        'div[contains(@class,"d_post_content")]//cc/div[@class="d_post_content j_d_post_content "]/text()')
                    if len(need) != 0:
                        item_dict['content'] = need[0][20:]
                    else:
                        item_dict['content'] = ''

                    need = floor.xpath(
                        'div[contains(@class,"d_post_content")]//div[@class="core_reply_tail clearfix"]/div[@class="post-tail-wrap"]/span[contains(text(),"-")]/text()')
                    if len(need) != 0:
                        item_dict['date'] = need[0]
                    else:
                        item_dict['date'] = ''

                    need = floor.xpath(
                        'div[contains(@class,"d_post_content")]//div[@class="core_reply_tail clearfix"]/div[@class="post-tail-wrap"]/span[contains(text(),"楼")]/text()')
                    if len(need) != 0:
                        item_dict['floor'] = need[0]
                    else:
                        item_dict['floor'] = ''

                    need = floor.xpath(
                        'div[contains(@class,"d_post_content")]//div[@class="core_reply_tail clearfix"]/div[@class="post-tail-wrap"]/span[contains(text(),"IP")]/text()')
                    if len(need) != 0:
                        item_dict['ip_address'] = need[0]
                    else:
                        item_dict['ip_address'] = ''

                    with open(new_path, 'a', encoding='utf-8') as fp:
                        fp.write(str(num))
                        for x in item_dict:
                            fp.write(',')
                            fp.write(item_dict[x])
                        fp.write('\n')
                    print('第%d层楼' % num)
                    num += 1
                # 翻页
                if flag == 1:
                    return
                try:
                    tie_page_button = driver.find_element(By.XPATH,
                                                          '//li[@class="l_pager pager_theme_4 pb_list_pager"]/a[text()="下一页"]')
                    tie_page_button.click()
                    print('下一页')
                except:
                    flag = 1
        except:
            traceback.print_exc()
            return

    def turn_page(self, driver):
        time.sleep(2)
        if '百度安全验证' in driver.page_source:
            a = my_web()
            a.main(driver)

        js_down = "window.scrollTo(0, 1000)"
        driver.execute_script(js_down)
        time.sleep(1)
        try:
            ba_page_button = driver.find_element(By.XPATH,
                                                 '//div[@id="frs_list_pager"]/a[@class="next pagination-item "]')
            ba_page_button.click()
            return 0
        except:
            return 1

    def run(self):

        try:
            status = 0
            driver = webdriver.Firefox()
            driver.get(self.page_url.format(self.kw,50))
            # driver.get(self.ba_url.format(self.kw))
            time.sleep(1)
            if '百度安全验证' in driver.page_source:
                a = my_web()
                a.main(driver)
            self.get_ba_content(driver)

            while (1):
                # 查找帖子列表
                tie_list = driver.find_elements(By.XPATH,
                                                '//li[@class=" j_thread_list clearfix thread_item_box"]//a[@class="j_th_tit "]')
                # 存储原始窗口的 ID
                original_window = driver.current_window_handle
                for i in range(0, len(tie_list)):
                    tie_list1 = driver.find_elements(By.XPATH,
                                                     '//li[@class=" j_thread_list clearfix thread_item_box"]//a[@class="j_th_tit "]')
                    # 点击进入帖子
                    print(f'第{i + 1}个帖子')
                    # 获取帖子链接
                    href = tie_list1[i].get_attribute('href')
                    # 在新的标签页打开链接
                    driver.execute_script(f'window.open("{href}", "_blank");')
                    # 切换到新的标签页
                    driver.switch_to.window(driver.window_handles[-1])
                    self.get_tie_content(driver)
                    # 关闭当前标签页
                    driver.close()

                    # 切回到之前的标签页
                    driver.switch_to.window(original_window)
                if status == 1:
                    break
                status = self.turn_page(driver)

            driver.quit()

        except Exception as e:
            with open('test.html', 'w', encoding='utf-8') as ep:
                ep.write(self.last_text)
            traceback.print_exc()


if __name__ == '__main__':
    spider = TiebaSpider("原神内鬼")
    spider.run()
