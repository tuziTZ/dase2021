import cv2
from selenium.webdriver.common.by import By
import random
from selenium.webdriver import ActionChains

from selenium import webdriver
import time
from lxml import etree
import requests
from selenium.webdriver.chrome.options import Options

# import my_ocr
from spin_img import my_ocr


class my_web():
    def __init__(self):
        option = Options()

        option.add_experimental_option('excludeSwitches', ['enable-automation'])
        option.add_argument('--disable-blink-features=AutomationControlled')


    def __ease_out_expo(self,sep):
        if sep == 1:
            return 1
        else:
            return 1 - pow(2, -10 * sep)


    def generate_tracks(self, distance):
        """
        根据滑动距离生成滑动轨迹
        :param distance: 需要滑动的距离
        :return: 滑动轨迹<type 'list'>: [[x,y,t], ...]
            x: 已滑动的横向距离
            y: 已滑动的纵向距离, 除起点外, 均为0
            t: 滑动过程消耗的时间, 单位: 毫秒
        """
        distance = int(distance)
        if not isinstance(distance, int) or distance < 0:
            raise ValueError(f"distance类型必须是大于等于0的整数: distance: {distance}, type: {type(distance)}")
        # 初始化轨迹列表
        slide_track = [
            # [random.randint(-50, -10), random.randint(-50, -10), 0],
            [0, 0, 0],
        ]
        # 共记录count次滑块位置信息
        count = 30 + int(distance / 2)
        # 初始化滑动时间
        t = random.randint(50, 100)
        # 记录上一次滑动的距离
        _x = 0
        _y = 0
        for i in range(count):
            # 已滑动的横向距离
            x = round(self.__ease_out_expo(i / count) * distance)
            # 滑动过程消耗的时间
            t += random.randint(10, 20)
            if x == _x:
                continue
            slide_track.append([x, _y, t])
            _x = x
        slide_track.append(slide_track[-1])
        return slide_track


    def main(self,driver):
        ocr = my_ocr()
        while True:
            # 打开网页1
            time.sleep(3)
            html = driver.page_source
            # print(html)
            html = etree.HTML(html)
            url = html.xpath('//*[@class="vcode-spin-img"]/@src')[0]
            response = requests.get(url).content
            with open('1.png', 'wb')as f:
                f.write(response)

            result = ocr.identification('1.png')
            displacement_distance = 212 / 360 * int(result)
            print('预测旋转角度为：',result,'滑动距离为：',displacement_distance)
            source = driver.find_element(By.XPATH, r'//*[@class="vcode-spin-button"]/p')
            action = ActionChains(driver, duration=10)
            action.click_and_hold(source).perform()
            a = 0
            for x in self.generate_tracks(displacement_distance):
                # time.sleep(random.uniform(0.1,0.2))
                # print(x)
                action.move_by_offset(xoffset=x[0]-a, yoffset=x[1])
                a = x[0]

            action.release(source).perform()
            # ActionChains(self.driver).drag_and_drop_by_offset(b, xoffset=displacement_distance, yoffset=0).perform()
            time.sleep(2)
            if '百度安全验证' not in driver.page_source:
                break
            # else:
            #
            #     with open('./{}.png'.format(int(time.time())), 'wb')as f:
            #         f.write(response)

# if __name__ == '__main__':
#     a = my_web()
#     a.main(driver)
