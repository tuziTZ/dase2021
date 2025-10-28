#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 数据集组织形式：
# 吧信息文档：id,name,card_slogan,关注,帖子,目录
# 主题帖信息文档：id,href,title,lz_name,lz_href
# 回帖信息文档：id,author_name,author_level,content(emoji=num),ip,time
# 楼中楼信息文档：id,author_name,content,time
import traceback
import requests
from lxml import etree
import json
import re
import os
from selenium import webdriver
import time


class TiebaSpider:

    def __init__(self, kw):
        self.kw = kw
        self.ba_url = "https://tieba.baidu.com/f?kw={}&ie=utf-8&pn={}"
        self.headers = {
            "Cookie": 'BAIDUID=4C55030C93817121CD4173006EBFBFAE:FG=1; BIDUPSID=4C55030C93817121820F688D61D071C2; '
                      'PSTM=1670744861; BDORZ=FFFB88E999055A3F8A630C64834BD6D0; '
                      'Hm_lvt_98b9d8c2fd6608d564bf2ac2ae642948=1670771825,1671199892,1671264355,1671347157; '
                      'BDUSS=Ww5cm1DRTVvVnA2VmRuT1cxN01ER3lpMjB2Yy1WWFlSWVd4dDVBYWM4S2M4c1JqRVFBQUFBJCQAAAAAAAAAAAEAAABkdfhT0KHQptCmX0VuZGVyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJxlnWOcZZ1jT; STOKEN=32eabe27ddf8a1788bf6145125e719cd72e3ce3af8ff46113baee2bee60ab4f2; H_PS_PSSID=37857_36561_37521_37906_37923_37951_37900_26350_37785_37881; delPer=0; PSINO=2; BA_HECTOR=0g2484ak8g8k8kak00ah2arr1hptef31h; ZFY=cZLqjkcHy7fc9lTI7lgtyC3CNwROdZeeuJAGCySAiZk:C; tb_as_data=0893550a8504d3335ef875dfe554335f68688ef8ad140231aae25d39881b537a09121511eb55c1eb72e55d3e6d6a514a9e39e307652814a8d8e1b3d9e72121c4f4b21b9ab2aaa9502634df416f8b75e6331ae17f299e274705cd46e0f48ae704838f7d9bf4e8b0fd0aba59c3694c0bf4; RT="z=1&dm=baidu.com&si=b7777ed1-0ea0-46a0-b2b9-be5416f55d77&ss=lbt0xzt5&sl=x&tt=16ws&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&ld=8i6u7&ul=8ia8k"; Hm_lpvt_98b9d8c2fd6608d564bf2ac2ae642948=1671361438; BAIDU_WISE_UID=wapp_1671347157547_584; USER_JUMP=-1; XFI=ac014400-7ec3-11ed-89ea-8bc23a663be2; XFCS=A2EB3EF9633DFDDBE882027CB8025A8391320405613BED7D6F22C26BAC15E3E7; st_data=29e6c0bbfbda3a39cfa9949802607c75692cc7c32bded9dc7b1013b1a94dd322c4c19eef6ee177f4b28861507788cf01c7babc0c3af973670983a5eeb000617469ebb6859bb9670d0c58d484fc76785bbe98e69d8d528a98e7ac52e1bafa4174; st_key_id=17; st_sign=cc1702fd; XFT=vq3IOYUa0OsEdJbue+iFsampSlJ4No9cK+Umd4xDb7E=; __bid_n=1851eb226b3e305c024207; FEID=v10-7951235f2e9424479ca8e783b9ad63104d53b496; __xaf_fpstarttimer__=1671347812664; __xaf_ths__={"data":{"0":1,"1":43200,"2":60},"id":"177c07de-4028-4f3d-a214-7c45905d33ce"}; __xaf_thstime__=1671347813136; FPTOKEN=TVV/5CgY4TQ0/CDEqJQ8FiJRLB+ZcGaMoxUYKgX3OhiPyK5mN+Gdi1fYe2ye/4XEMTFVSJdTHhyHPlHpGzG0MiMBxQO0pPoPBjhLi9dcfO6+0vYHy+hqxa2rCXTJguwdyIYIJ3bK6XckkweYYz4MjfUtjvVoomi+u0lkQ6DUm9HzHeT5gbgS4EvaPKRXt9XbCNQBTVk4JPlphN0AoisVv23tZ2NnUhjoDWW8Q5bwfnn3ZNmuXWWQRCboRqipiwwsJLqtF+pJVbCjMlvbL/OiS0i5SvHNpjYD3s7fVEF0azOG8OVRicYH9l25zHFz/17p/0hdXR1kF/SHJTqYo7+9uFkZyOe+Lf8HUh+jfQ8lpdA7OaaNDNJDVWOwgfCjNwynq/O1tiErmW3aoE5nHU9EMw==|5beX+SWnYzwXHliyTC/TtHSuxgSEfYBGhjjEUgfectA=|10|cd82d7348edf842a8731708f01656a7a; __xaf_fptokentimer__=1671347813138; wise_device=0; ab_sr=1.0.1_NDFhZjI1NjkzNDUxZWQxODAxNTMyNDNlNjY0ZTM0ZThjMDIyM2M4MDgxM2FhZWQ5OTM0ZGMzZDVjNzcxNzBkN2RhMTUwZGY1OTNmNDQwYTU0OWIzMDc0NTNiZTljMjczNzBlM2ZmM2NiNTRjNGJhZTk5NGYyYzU3OTcyMTQ2YjVlYjRlYzEzODhjMmY0OTE1OTIzMjhhZTBhZjM2N2VjODE3Y2VjMGRlOTRlNzIxNDhiOTUwZTM1YjRiZGNkZDk2; BDRCVFR[Fc9oatPmwxn]=aeXf-1x8UdYcs; ZD_ENTRY=baidu; BDRCVFR[gZhL2P1o08b]=mk3SLVN4HKm; BCLID=7264920386678813415; BCLID_BFESS=7264920386678813415; BDSFRCVID=_LLOJeCT5G09-d6jwwmEeljIoyxIU97TTPjcTR5qJ04BtyCVcmiREG0Ptsp1nZLM_EGSogKKymOTHrAF_2uxOjjg8UtVJeC6EG0Ptf8g0M5; BDSFRCVID_BFESS=_LLOJeCT5G09-d6jwwmEeljIoyxIU97TTPjcTR5qJ04BtyCVcmiREG0Ptsp1nZLM_EGSogKKymOTHrAF_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF=tbIJoDK5JDD3fP36q45HMt00qxby26nHamc9aJ5nQI5nhKIzb5j85n-FKMoQafcyM6bA-CI5QUbmjRO206oay6O3LlO83h52aC5NKl0MLPb5qKjkWxvYBUL10UnMBMn8amOnaPop3fAKftnOM46JehL3346-35543bRTLnLy5KJYMDFRjj8KjjbBDHRf-b-XKD600PK8Kb7Vbp7xQxnkbft7jttjqCrbJDQfLM_h5RR8OJ7w2Mrlej-73b3B5h3NJ66ZoIbPbPTTSROzMq5pQT8r5hbjJ46zBgjdKl3Rab3vOPTzXpO1Kx_zBN5thURB2DkO-4bCWJ5TMl5jDh3Mb6ksD-FtqjttJnut_KLhf-3bfTrP-trf5DCShUFsyM7rB2Q-5M-a3KtBKJ-CMRJbhfL8hM6A2bcOWGI8_MbmLncjSM_GKfC2jMD32tbp5-r0LeTxoUJ2--taDxcsXqnpQptebPRih4r9QgbH5lQ7tt5W8ncFbT7l5hKpbt-q0x-jLTnhVn0MBCK0HPonHjDMD6JB3f; H_BDCLCKID_SF_BFESS=tbIJoDK5JDD3fP36q45HMt00qxby26nHamc9aJ5nQI5nhKIzb5j85n-FKMoQafcyM6bA-CI5QUbmjRO206oay6O3LlO83h52aC5NKl0MLPb5qKjkWxvYBUL10UnMBMn8amOnaPop3fAKftnOM46JehL3346-35543bRTLnLy5KJYMDFRjj8KjjbBDHRf-b-XKD600PK8Kb7Vbp7xQxnkbft7jttjqCrbJDQfLM_h5RR8OJ7w2Mrlej-73b3B5h3NJ66ZoIbPbPTTSROzMq5pQT8r5hbjJ46zBgjdKl3Rab3vOPTzXpO1Kx_zBN5thURB2DkO-4bCWJ5TMl5jDh3Mb6ksD-FtqjttJnut_KLhf-3bfTrP-trf5DCShUFsyM7rB2Q-5M-a3KtBKJ-CMRJbhfL8hM6A2bcOWGI8_MbmLncjSM_GKfC2jMD32tbp5-r0LeTxoUJ2--taDxcsXqnpQptebPRih4r9QgbH5lQ7tt5W8ncFbT7l5hKpbt-q0x-jLTnhVn0MBCK0HPonHjDMD6JB3f; 1408791908_FRSVideoUploadTip=1; video_bubble1408791908=1 '
            ,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36"
            , 'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9'
        }
        file_list = os.listdir('./tables/')
        self.max = 0
        self.last_text = ''
        for i in file_list:
            # print(i)
            if re.match(r"\d", i) != None:
                if self.max < (int)(i):
                    self.max = (int)(i)
                    # print(self.max)
        self.max += 1

        # self.browser = webdriver.Firefox()
        # dirver = webdriver.Firefox()
        # dirver.get('https://tieba.baidu.com/f?kw=%E5%AD%99%E7%AC%91%E5%B7%9D&ie=utf-8')
        # dictCookies = dirver.get_cookies()  # 获得所有cookie信息(返回是字典)
        #
        # jsonCookies = json.dumps(dictCookies)  # dumps是将dict转化成str格式
        # time.sleep(30)
        # dirver.quit()
        # # 登录完成后,将cookies保存到本地文件
        # with open("cookies_fofa.json", "w") as fp:
        #     fp.write(jsonCookies)

        pass

    def get_cookie(self):
        dirver = webdriver.Firefox()
        dirver.get('https://tieba.baidu.com/f?kw=%E5%AD%99%E7%AC%91%E5%B7%9D&ie=utf-8')
        dictCookies = dirver.get_cookies()  # 获得所有cookie信息(返回是字典)

        jsonCookies = json.dumps(dictCookies)  # dumps是将dict转化成str格式

        # 登录完成后,将cookies保存到本地文件
        with open("cookies.json", "w") as fp:
            fp.write(jsonCookies)

    def get_page_url(self):
        # 获取吧的基本信息:主题帖、关注、帖子
        url = 'https://tieba.baidu.com/f?kw={}&ie=utf-8'.format(self.kw)
        # self.get_cookie()
        # self.browser.get(url)
        # self.browser.delete_all_cookies()  # 删除所有cookie信息
        # with open('cookies_fofa.json', 'r', encoding='utf-8') as f:
        #     listCookies = json.loads(f.read())  # loads是将str转化成dict格式
        # for cookie in listCookies:
        #     self.browser.add_cookie(cookie)
        # self.browser.get(url)
        # page_text = self.browser.page_source
        response = requests.get(
            url=url,
            headers=self.headers
        )
        page_text = response.text
        self.last_text = page_text
        ex = '<!--(.*?)-->'
        print(url)
        # text=page_text
        # print(re.findall(ex, page_text, re.S))
        if len(re.findall(ex, page_text, re.S)) < 47:
            text = page_text
        else:
            text = re.findall(ex, page_text, re.S)[48]
        self.last_text = text

        with open('./tables/%s.txt' % self.kw, 'w', encoding='utf-8') as bafp:
            tree = etree.HTML(text)
            num = tree.xpath('//div[@class="th_footer_l"]/span/text()')
            for i in num:
                bafp.write(i + ' ')
            bafp.write(str(self.max) + '\n')
            max_pn = (int)(num[0])

        # max_pn+=50
        max_pn = 10000

        page_url = []
        for pn in range(0, max_pn, 50):
            url = self.ba_url.format(self.kw, pn)
            page_url.append(url)
        with open('page_url.txt', 'w', encoding='utf-8') as fp:
            for i in page_url:
                fp.write(i + '\n')
        return page_url

    def get_ba_content(self, page_url):
        sel_list = []
        flag = 0
        for url in page_url:
            print(url)
            response = requests.get(
                url=url,
                headers=self.headers
            )
            page_text = response.text
            self.last_text = page_text
            # self.browser.get(url)
            # page_text = self.browser.page_source
            # text = page_text
            ex = '<!--(.*?)-->'
            #
            # self.get_cookie()
            # self.browser.get(url)
            # self.browser.delete_all_cookies()  # 删除所有cookie信息
            # with open('cookies_fofa.json', 'r', encoding='utf-8') as f:
            #     listCookies = json.loads(f.read())  # loads是将str转化成dict格式
            # for cookie in listCookies:
            #     self.browser.add_cookie(cookie)
            # self.browser.get(url)
            # page_text = self.browser.page_source
            # 获取每个主题帖的url
            # text = page_text
            re_list = re.findall(ex, page_text, re.S)
            if len(re_list) < 47:
                text = page_text
            else:
                max = 0
                set = ''
                for i in re_list:
                    if max < len(i):
                        max = len(i)
                        set = i
                text = set
            self.last_text = text

            tree = etree.HTML(text)
            if flag == 0:
                flag = 1
            time.sleep(2)
            sel = tree.xpath('//li[@class=" j_thread_list clearfix thread_item_box"]//a[@class="j_th_tit "]/@href')
            sel_list += sel
            print(sel[0])

            with open('sel_list.txt', 'a', encoding='utf-8') as fp:
                for i in sel:
                    fp.write(i + '\n')
        return sel_list

    def get_tie_content(self, sel_list):
        url = 'https://tieba.baidu.com'

        for join in sel_list:

            response = requests.get(
                url=url + join,
                headers=self.headers
            )
            page_text = response.text
            self.last_text = page_text
            # self.browser.get(url)
            # page_text = self.browser.page_source
            #
            # ex = '<!--(.*?)-->'
            # if len(re.findall(ex, page_text, re.S)) < 47:
            #     self.get_cookie()
            #     self.browser.get(url)
            #     self.browser.delete_all_cookies()  # 删除所有cookie信息
            #     with open('cookies_fofa.json', 'r', encoding='utf-8') as f:
            #         listCookies = json.loads(f.read())  # loads是将str转化成dict格式
            #     for cookie in listCookies:
            #         self.browser.add_cookie(cookie)
            #     self.browser.get(url)
            #     page_text = self.browser.page_source

            tree = etree.HTML(page_text)
            if tree.xpath('head/title/text()') == '贴吧404':
                continue
            title = tree.xpath('//h3[contains(@class,"core_title_txt pull-left text-overflow  ")]/@title')[0]
            pn = tree.xpath('//span[@class="red"]/text()')[1]

            # href_list=[]
            # author_list=[]
            # level_list = []
            # new_content_list=[]
            # date_list=[]
            # ip_list=[]

            print((int)(pn) + 1)

            path = './tables/%s' % str(self.max)
            if os.path.exists(path) == 0:
                os.mkdir(path)
            with open(path + '/' + str(sel_list.index(join)) + '.txt', 'w', encoding='utf-8') as fp:
                fp.write(title + '\n')
                fp.write('id,user_href,user_level,is_lz,user_name,content,date,floor,ip_address\n')

                num = 1
                for i in range(1, (int)(pn) + 1):
                    # for i in range(1,2):
                    time.sleep(2)

                    # self.get_cookie()
                    # self.browser.get(url)
                    # self.browser.delete_all_cookies()  # 删除所有cookie信息
                    # with open('cookies_fofa.json', 'r', encoding='utf-8') as f:
                    #     listCookies = json.loads(f.read())  # loads是将str转化成dict格式
                    # for cookie in listCookies:
                    #     self.browser.add_cookie(cookie)
                    # self.browser.get(url)
                    # page_text = self.browser.page_source
                    response = requests.get(
                        url=url + join + '?pn=' + str(i),
                        headers=self.headers
                    )
                    page_text = response.text
                    self.last_text = page_text

                    tree = etree.HTML(page_text)
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

                        need = floor.xpath(
                            'div[@class="d_author"]/ul[@class="p_author"]//div[@class="d_badge_lv"]/text()')
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

                        # href_list1=tree.xpath('//div[@class="l_post l_post_bright j_l_post clearfix  "]//a[@class="p_author_name j_user_card"]/@href')
                        # author_list1=tree.xpath('//div[@class="l_post l_post_bright j_l_post clearfix  "]//a[@class="p_author_name j_user_card"]/text()')
                        # level_list1=tree.xpath('//div[@class="l_post l_post_bright j_l_post clearfix  "]//div[@class="d_badge_lv"]/text()')
                        # content_list1=tree.xpath('//div[@class="l_post l_post_bright j_l_post clearfix  "]//cc/div[@class="d_post_content j_d_post_content "]/text()')
                        # print(level_list1)
                        #
                        #
                        # new_content_list1=[]
                        # for i in content_list1:
                        #     new_content_list1.append(i[20:])
                        #
                        #
                        # span_list1=tree.xpath('//div[@class="l_post l_post_bright j_l_post clearfix  "]//span[@class="tail-info"]/text()')
                        # date_list1=[]
                        # for i in range(2,len(span_list1),3):
                        #     date_list1.append(span_list1[i])
                        #
                        # ipaddress_list1=tree.xpath('//div[@class="l_post l_post_bright j_l_post clearfix  "]//div[@class="post-tail-wrap"]/span/text()')
                        # ip_list1=[]
                        # for i in range(0,len(ipaddress_list1),4):
                        #     ip_list1.append(ipaddress_list1[i][-2:])
                        #
                        # # print(len(href_list1),len(author_list1),len(new_content_list1),len(date_list1),len(ip_list1))
                        # href_list += href_list1
                        # author_list += author_list1
                        # new_content_list += new_content_list1
                        # date_list += date_list1
                        # ip_list += ip_list1
                        # print('1')
                        fp.write(str(num))
                        for x in item_dict:
                            fp.write(',')
                            fp.write(item_dict[x])
                        fp.write('\n')
                        num += 1

            print(sel_list.index(join))

    def run(self):
        # 1. 获取每一页的url
        # page_url = self.get_page_url()

        try:
            with open('page_url.txt', 'r', encoding='utf-8') as fp:
                line = fp.readline()[:-1]
                while (line):
                    page_url = []
                    page_url.append(line)
                    line = fp.readline()[:-1]
                    sel_list = self.get_ba_content(page_url)
                    self.get_tie_content(sel_list)
        except Exception as e:
            with open('test.html', 'w', encoding='utf-8') as ep:
                ep.write(self.last_text)
            traceback.print_exc()
        # self.get_cookie()


if __name__ == '__main__':
    spider = TiebaSpider("孙笑川")
    spider.run()
