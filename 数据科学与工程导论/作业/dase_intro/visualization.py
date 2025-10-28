#!/usr/bin/python3
# -*- coding: utf-8 -*-

import traceback
import pandas as pd
import json
import re
import time
import datetime
import os
import pyecharts
import pretreat
import random
import jieba
import jieba.posseg as pseg
from pyecharts.charts import Map
from pyecharts import options as opts
from pyecharts.faker import Faker
from pyecharts.charts import Calendar
from pyecharts.commons.utils import JsCode
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import snownlp

fn = """
    function(params) {
        if(params.name == '其他')
            return '\\n\\n\\n' + params.name + ' : ' + params.value + '%';
        return params.name + ' : ' + params.value + '%';
    }
    """


def Q1(ba_list):
    bar = pyecharts.charts.Bar(init_opts=opts.InitOpts(theme='light',
                                                       width='1000px',
                                                       height='600px'))
    df = pretreat.combine(ba_list[0])
    sequence = pretreat.ba_ip(df)
    data = list(sequence)

    data.sort(key=lambda tup: tup[1], reverse=True)
    data = data[:7]
    data = list(zip(*data))
    province_list = data[0]
    value1 = list(data[1])
    value2 = []
    df = pretreat.combine(ba_list[1])
    sequence = pretreat.ba_ip(df)
    data = list(sequence)
    for i in province_list:
        for j in data:
            if j[0] == i:
                value2.append(j[1])
                break

    bar.add_xaxis(province_list)
    bar.add_yaxis(ba_list[0], value1)
    bar.add_yaxis(ba_list[1], value2)
    bar.set_global_opts(title_opts=opts.TitleOpts(title="最多用户发言省区Top6"))
    bar.render(path="Q1.html")

    for ba_name in ba_list:
        df = pretreat.combine(ba_name)
        sequence = pretreat.ba_ip(df)
        data = list(sequence)

        data.sort(key=lambda tup: tup[1], reverse=True)
        max1 = data[0][1]
        map = Map(opts.InitOpts(width='1200px', height='600px'))

        map.add(maptype='china', data_pair=sequence, series_name=ba_name).set_global_opts(
            title_opts=opts.TitleOpts(title="每省发言用户数量"),
            visualmap_opts=opts.VisualMapOpts(max_=max1, min_=0)

        )
        map.label_opts = opts.LabelOpts(is_show=True)

        map.render(path="Q1_" + str(ba_list.index(ba_name)) + ".html")


def Q2(ba_name):
    df = pretreat.combine(ba_name)
    sequence = pretreat.ip_fre(df)

    map = Map(opts.InitOpts(width='1200px', height='600px'))

    map.add(maptype='china', data_pair=sequence, series_name=ba_name).set_global_opts(
        title_opts=opts.TitleOpts(title="省平均用户发言数"),
        visualmap_opts=opts.VisualMapOpts(max_=2.5, min_=1.5),

    )

    map.render(path="Q2_2.html")


def Q3(ba_list):
    fre_list = []
    for i in ba_list:
        fre_list.append(pretreat.tie_delete(i))

    c = pyecharts.charts.Boxplot(init_opts=opts.InitOpts(width="800px", height="500px"))  # 设置大小
    c.add_xaxis(ba_list)

    c.add_yaxis('每楼删帖率', c.prepare_data(fre_list))

    c.set_global_opts(title_opts=opts.TitleOpts(title="吧内删帖率"))
    c.render("Q3_box.html")


def Q4_1(ba_list):
    pie = pyecharts.charts.Pie(init_opts=opts.InitOpts(theme='sex',
                                                       width='1000px',
                                                       height='600px'))
    width = 60 / len(ba_list)
    for i in range(0, len(ba_list)):
        df = pretreat.wash_user_table(ba_list[i])

        x_data, y_data = pretreat.sex(df)
        x_data_ = ['', '']
        x_data_[0] = x_data[0] + '-' + ba_list[i] + '吧'
        x_data_[1] = x_data[1] + '-' + ba_list[i] + '吧'
        print(x_data_)
        r1 = 20 + i * width
        r2 = r1 + width - 5

        pie.add(series_name=ba_list[i],
                data_pair=[list(z) for z in zip(x_data_, y_data)],
                # 设置半径范围，0%-100%
                radius=[str(r1) + '%', str(r2) + '%'],
                center=["50%", "50%"],
                label_opts=opts.LabelOpts(is_show=True, position='inner'),

                )
    pie.set_global_opts(title_opts=opts.TitleOpts(title="性别比例"))
    pie.render(path='Q4_1.html')


def Q4_2(ba_list):
    bar = pyecharts.charts.Bar(init_opts=opts.InitOpts(theme='light',
                                                       width='1000px',
                                                       height='600px'))
    df = pretreat.wash_user_table(ba_list[0])
    attr, value = pretreat.age(df)

    bar.add_xaxis(attr)
    for ba_name in ba_list:
        df = pretreat.wash_user_table(ba_name)
        attr, value = pretreat.age(df)

        bar.add_yaxis(ba_name, value)
    bar.set_global_opts(title_opts=opts.TitleOpts(title="吧龄分布"))
    bar.render(path="Q4_2.html")


def Q4_2_box(ba_list):
    c = pyecharts.charts.Boxplot(init_opts=opts.InitOpts(width="800px", height="500px"))  # 设置大小
    data_list = []
    c.add_xaxis(ba_list)
    for i in ba_list:
        df = pretreat.wash_user_table(i)
        ba_age_list = list(df['age'].dropna())
        print(ba_age_list)
        data_list.append(ba_age_list)

    c.add_yaxis('吧龄分布', c.prepare_data(data_list))

    c.set_global_opts(title_opts=opts.TitleOpts(title="吧龄分布"))
    c.render("Q4_2_box.html")


def Q5(ba_list):
    c = pyecharts.charts.Boxplot(init_opts=opts.InitOpts(width="800px", height="500px"))  # 设置大小
    data_list = []
    c.add_xaxis(ba_list)
    for i in ba_list:

        percent_list = []
        txt_list = os.listdir('tables/' + i)
        for j in range(0, len(txt_list)):
            df = pretreat.wash_tie_table(i, j)
            percent = pretreat.lz(df)
            if percent != 0:
                percent_list.append(float(percent))
        data_list.append(percent_list)
    c.add_yaxis('楼主发言占比', c.prepare_data(data_list))

    c.set_global_opts(title_opts=opts.TitleOpts(title="楼主发言占比"))
    c.render("Q5.html")


def Q5_1(ba_list):
    pie = pyecharts.charts.Pie(init_opts=opts.InitOpts(width='1000px',
                                                       height='600px'))
    width = 60 / len(ba_list)

    lz_list = ['楼主发言', '非楼主发言']
    for i in ba_list:

        percent_list = []
        txt_list = os.listdir('tables/' + i)
        for j in range(0, len(txt_list)):
            df = pretreat.wash_tie_table(i, j)
            percent = pretreat.lz(df)
            if percent != 0:
                percent_list.append(float(percent))
        p_list = [sum(percent_list) / len(percent_list), 1 - sum(percent_list) / len(percent_list)]
        r1 = 20 + ba_list.index(i) * width
        r2 = r1 + width - 5
        lz_list_ = ['', '']
        lz_list_[0] = lz_list[0] + (':' + i + '吧')
        lz_list_[1] = lz_list[1] + (':' + i + '吧')
        pie.add(series_name=i,
                data_pair=[list(z) for z in zip(lz_list_, p_list)],
                # 设置半径范围，0%-100%
                radius=[str(r1) + '%', str(r2) + '%'],
                center=["50%", "50%"],
                label_opts=opts.LabelOpts(position='inner')
                )
        pie.set_global_opts(title_opts=opts.TitleOpts(title="楼主发言占比"))
    pie.render(path='Q5_1.html')


def Q6(ba_name):
    df = pretreat.combine(ba_name)
    attr, value, act = pretreat.level(df)
    attr = list(map(str, attr))
    mid_np = np.array(act)
    mid_np_2f = np.round(mid_np, 2)

    pie = pyecharts.charts.Pie(init_opts=opts.InitOpts(theme='等级分布',
                                                       width='1000px',
                                                       height='600px'))

    bar = (
        pyecharts.charts.Bar()
        .add_xaxis(attr)
        .add_yaxis('人数', value, color='rgba(160, 23, 137,0.5)')
        .extend_axis(
            yaxis=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}条"), min_=1
            )
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=ba_name + "吧\n等级与发言数"),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}人")),
        )
    )
    act = list(mid_np_2f)
    line = pyecharts.charts.Line().add_xaxis(attr).add_yaxis('平均每人发言数', act, yaxis_index=1)
    bar.overlap(line)  # 在柱状图上叠加折线图
    bar.render(path="Q6_2.html")

    pie.add("",
            data_pair=[list(z) for z in zip(attr, value)],
            # 设置半径范围，0%-100%
            radius=["40%", "75%"],
            center=["50%", "50%"],
            label_opts=opts.LabelOpts(
                position="outside",
                formatter="{d}%"
            ))
    pie.set_global_opts(
        title_opts=opts.TitleOpts(title=ba_name + "吧等级分布"),
    )

    pie.render(path='Q6_1.html')


def Q7(ba_name):
    data = pretreat.date(ba_name)
    c = (
        Calendar()
        .add("", data, calendar_opts=opts.CalendarOpts(
            range_=["2022-3-1", '2022-12-20'],
        ),
             )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=ba_name + "吧-2022年发帖情况"),
            visualmap_opts=opts.VisualMapOpts(
                orient="horizontal",
                is_piecewise=True,
                pos_top="230px",
                pos_left="10px",
            ),
        )
        .render("Q7.html")
    )


def Q8(ba_name):
    df = pretreat.combine(ba_name)
    time_index, time_table = pretreat.time_(df)
    line = pyecharts.charts.Line()
    line.add_xaxis(time_table)
    line.add_yaxis("发言条数", time_index)

    line.set_global_opts(title_opts=opts.TitleOpts(title="Line-基本示例"))
    line.render("Q8.html")


def Q8_(ba_list):
    df = pretreat.combine(ba_list[0])
    time_index, time_table = pretreat.time_(df)
    line = pyecharts.charts.Line()
    line.add_xaxis(time_table)
    for i in ba_list:
        df = pretreat.combine(i)
        time_index, time_table = pretreat.time_(df)
        line.add_yaxis(i, time_index)

    line.set_global_opts(title_opts=opts.TitleOpts(title="各吧发言活跃时段"))
    line.render("Q8.html")


def Q9(ba_name):
    df = pretreat.combine(ba_name)
    dataset_new, dataset_old = pretreat.emo(df)
    data = list(zip(*dataset_new))
    y_data = data[1][-10:]
    x_data = data[0][-10:]
    y2_data = list(zip(*dataset_old))[1][-10:]
    bar = pyecharts.charts.Bar(init_opts=opts.InitOpts(width="800px", height="600px"))
    bar.add_xaxis(x_data)
    bar.add_yaxis('emoji_old', y2_data, stack='stack1')
    bar.add_yaxis('emoji_new', y_data, stack='stack1')
    bar.set_global_opts(title_opts=opts.TitleOpts(title=ba_name + '吧用户最常用表情'),
                        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=13, rotate=15)))
    bar.reversal_axis()
    bar.render("Q9.html")


def Q9_(ba_name):
    df = pretreat.combine(ba_name)
    dataset = pretreat.emo_(df)
    data = list(zip(*dataset))
    y_data = data[1][-50:]
    x_data = data[0][-50:]
    dataset = list(zip(x_data, y_data))
    mywordcloud = pyecharts.charts.WordCloud(init_opts=opts.InitOpts(width="800px", height="500px"))
    mywordcloud.add('', dataset, shape='circle')
    mywordcloud.set_global_opts(title_opts=opts.TitleOpts(title=ba_name + "吧用户最常用表情"))
    mywordcloud.render("Q9_.html")


def Q10(ba_name):
    dataset = pretreat.hot_words(ba_name)
    data = list(zip(*dataset))
    y_data = data[1][-50:]
    x_data = data[0][-50:]
    dataset = list(zip(x_data, y_data))
    mywordcloud = pyecharts.charts.WordCloud(init_opts=opts.InitOpts(width="1000px", height="800px"))
    mywordcloud.add('', dataset, shape='circle')
    mywordcloud.set_global_opts(opts.TitleOpts(title=ba_name + '吧常用词汇'))
    mywordcloud.render("Q10.html")


def Q11(ba_name):
    df = pretreat.corr(ba_name)
    corrdf = df.corr()
    index = ['user_level', 'counts_lz', 'counts', 'ip_address', 'sex', 'age', 'num', 'follow_peo', 'follow_ba']
    key_list = list(corrdf)
    value = [[i, j, corrdf[i][j]] for i in key_list for j in range(0, len(key_list))]
    c = (
        pyecharts.charts.HeatMap()
        .add_xaxis(index)
        .add_yaxis("", index, value, label_opts=opts.LabelOpts(is_show=True, position="inside"), )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=ba_name + "吧用户指标相关矩阵热力图"),
            visualmap_opts=opts.VisualMapOpts(min_=-0.5, max_=0.5),
        )
        .render("Q11.html")
    )


def Q12(ba_name):
    x_data, y_data = pretreat.line(ba_name)
    scatter = pyecharts.charts.Scatter(init_opts=opts.InitOpts(width="800px", height="500px"))
    scatter.add_xaxis(xaxis_data=x_data)
    scatter.add_yaxis(
        series_name="",
        y_axis=y_data,
        symbol_size=4,
        label_opts=opts.LabelOpts(is_show=False),
    )
    scatter.set_series_opts()
    scatter.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
    )
    scatter.render(path='Q12.html')


def loadDict(fileName):
    wordDict = {}
    with open(fileName, encoding='utf-8') as fin:
        for line in fin:
            list_ = line.strip().split(' ')
            if len(list_) == 2:
                word = list_[0]
                wordDict[word] = float(list_[1])
    return wordDict


def loadDict_(fileName, score):
    wordDict = {}
    with open(fileName, encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            wordDict[word] = score
    return wordDict


def renewDict(wordDict, word, score):
    if word not in wordDict:
        wordDict[word] = score
    else:
        wordDict[word] = (0.9 * wordDict[word] + 0.1 * score)


def appendDict(wordDict, fileName):
    with open(fileName, 'w', encoding='utf-8') as fp:
        for word in wordDict:
            fp.write(word + ' ' + str(wordDict[word]) + '\n')


def loadExtentDict(fileName, level):
    extentDict = {}
    for i in range(level):
        with open(fileName + str(i + 1) + ".txt", encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                extentDict[word] = i + 1
    return extentDict


stopclass = ['uj', 'r', 'd', 'x', 's', 'p']


def EmoScore(ba_name):
    postDict = loadDict(u"正面情感词语.txt")
    negDict = loadDict(u"负面情感词语.txt")
    inverseDict = loadDict_(u"否定词.txt", -1)
    extentDict = loadExtentDict(u"程度级别词语", 6)
    exclamation = {"!": 2, "！": 2}
    punc = loadDict_(u"标点符号.txt", 1)

    df = pretreat.combine(ba_name)
    content_list = list(df['content'].astype(str))
    totalScore_list = []
    for content in content_list:
        words = pseg.cut(content)
        wordList_ = list(words)
        wordList = []
        renew_list = []
        for word, flag in wordList_:
            wordList.append(word)
            if flag not in stopclass:
                renew_list.append(word)

        totalScore = 0  # 记录最终情感得分
        lastWordPos = 0  # 记录情感词的位置
        lastPuncPos = 0
        i = 0  # 记录扫描到的词的位置
        start = 0
        for word in wordList:
            if word in punc:
                lastPuncPos = i
            if word in postDict:
                if word in postDict:
                    if lastWordPos > lastPuncPos:
                        start = lastWordPos
                    else:
                        start = lastPuncPos
                score = postDict[word]
                for word_before in wordList[start:i]:
                    if word_before in extentDict:
                        score = score * extentDict[word_before]
                    if word_before in inverseDict:
                        score = score * -1.0
                for word_after in wordList[i + 1:]:
                    if word_after in exclamation:
                        score = score + 2.0
                    else:
                        break
                lastWordPos = i

                totalScore += score
            elif word in negDict:
                if word in postDict:
                    if lastWordPos > lastPuncPos:
                        start = lastWordPos
                    else:
                        start = lastPuncPos
                score = negDict[word]
                for word_before in wordList[start:i]:
                    if word_before in extentDict:
                        score = score * extentDict[word_before]
                    if word_before in inverseDict:
                        score = score * -1.0
                for word_after in wordList[i + 1:]:
                    if word_after in exclamation:
                        score = score - 2.0
                    else:
                        break
                lastWordPos = i
                totalScore += score
            i += 1

        for word in renew_list:
            l = len(renew_list)
            score = totalScore / (l * l)
            if score < 0:
                renewDict(negDict, word, score)
            else:
                renewDict(postDict, word, score)

        totalScore_list.append(totalScore)
    appendDict(negDict, '负面情感词语-已训练.txt')
    appendDict(postDict, '正面情感词语-已训练.txt')
    return totalScore_list


def Q14(ba_name):
    scorelist = getScore(ba_name)
    x_data = list(range(0, len(scorelist)))
    y_data = list(zip(x_data, scorelist))
    line = pyecharts.charts.Line(init_opts=opts.InitOpts(width='900px', height='400px', theme='light'))
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis('',
                   y_axis=scorelist,
                   label_opts=opts.LabelOpts(is_show=False),
                   )
    line.set_global_opts(xaxis_opts=opts.AxisOpts(is_show=True),
                         yaxis_opts=opts.AxisOpts(is_show=True),
                         title_opts=opts.TitleOpts(title='情感评分'),
                         )
    line.render(path='Q14.html')


def testemo(content_list):
    postDict = loadDict(u"正面情感词语-已训练.txt")  # 积极情感词典
    negDict = loadDict(u"负面情感词语-已训练.txt")  # 消极情感词典
    inverseDict = loadDict_(u"否定词.txt", -1)  # 否定词词典
    extentDict = loadExtentDict(u"程度级别词语", 6)
    exclamation = {"!": 2, "！": 2}
    punc = loadDict_(u"标点符号.txt", 1)
    for content in content_list:
        print(content)
        words = jieba.cut(content)
        wordList = list(words)
        totalScore = 0  # 记录最终情感得分
        lastWordPos = 0  # 记录情感词的位置
        lastPuncPos = 0
        i = 0  # 记录扫描到的词的位置
        start = 0
        for word in wordList:
            if word in punc:
                lastPuncPos = i
            if word in postDict:
                if word in postDict:
                    if lastWordPos > lastPuncPos:
                        start = lastWordPos
                    else:
                        start = lastPuncPos
                score = postDict[word]
                for word_before in wordList[start:i]:
                    if word_before in extentDict:
                        score = score * extentDict[word_before]
                    if word_before in inverseDict:
                        score = score * -1.0
                for word_after in wordList[i + 1:]:
                    if word_after in exclamation:
                        score = score + 2.0
                    else:
                        break
                lastWordPos = i

                totalScore += score
            elif word in negDict:
                if word in postDict:
                    if lastWordPos > lastPuncPos:
                        start = lastWordPos
                    else:
                        start = lastPuncPos
                score = negDict[word]
                for word_before in wordList[start:i]:
                    if word_before in extentDict:
                        score = score * extentDict[word_before]
                    if word_before in inverseDict:
                        score = score * -1.0
                for word_after in wordList[i + 1:]:
                    if word_after in exclamation:
                        score = score - 2.0
                    else:
                        break
                lastWordPos = i
                totalScore += score
            i += 1
        print(totalScore)


def split_time_ranges(from_time, to_time, frequency):
    from_time, to_time = pd.to_datetime(from_time), pd.to_datetime(to_time)
    time_range = list(pd.date_range(from_time, to_time, freq='%sS' % frequency))
    if to_time not in time_range:
        time_range.append(to_time)
    time_range = [item.strftime("%Y-%m-%d %H:%M") for item in time_range]
    time_ranges = []
    for item in time_range:
        f_time = item
        t_time = (datetime.datetime.strptime(item, "%Y-%m-%d %H:%M") + datetime.timedelta(seconds=frequency))
        if t_time >= to_time:
            t_time = to_time.strftime("%Y-%m-%d %H:%M")
            time_ranges.append([f_time, t_time])
            break
        time_ranges.append([f_time, t_time.strftime("%Y-%m-%d %H:%M")])
    return time_ranges


def Q13(ba_name):
    obj_list = pretreat.pick_most_floor(ba_name)

    for i1 in range(0, len(obj_list)):
        df = obj_list[i1]
        df_list = list(df)
        from_time = df_list[0][:14] + '00'
        to_time = df_list[len(df_list) - 1][:14] + '00'
        frequency = 60 * 15
        time_ranges = split_time_ranges(from_time, to_time, frequency)
        value_list = [0] * len(time_ranges)
        j = 0
        for i in df_list:
            if (i <= time_ranges[j][1]) & (i >= time_ranges[j][0]):

                value_list[j] += 1
            else:
                while ((i <= time_ranges[j][1]) & (i >= time_ranges[j][0]) != 1):
                    j += 1
                    if j >= len(time_ranges):
                        break
                if j >= len(time_ranges):
                    break
                value_list[j] += 1
        bar = (
            pyecharts.charts.Bar()
            .add_xaxis(time_ranges)
            .add_yaxis('发言条数', value_list, category_gap=0)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=ba_name + '吧-帖子' + str(i1 + 1) + "\n每15分钟发言数")
            )
            .render(path='Q13_' + str(i1 + 1) + '.html')
        )


def Q13_(ba_name):
    obj_list = pretreat.pick_most_floor(ba_name)

    for i1 in range(0, len(obj_list)):
        df = obj_list[i1]
        df_list = list(df)
        from_time = df_list[0][:14] + '00'
        to_time = df_list[len(df_list) - 1][:14] + '00'
        frequency = 60 * 15
        time_ranges = split_time_ranges(from_time, to_time, frequency)
        value_list = [0] * len(time_ranges)
        j = 0
        for i in df_list:
            if (i <= time_ranges[j][1]) & (i >= time_ranges[j][0]):

                value_list[j] += 1
            else:
                while ((i <= time_ranges[j][1]) & (i >= time_ranges[j][0]) != 1):
                    j += 1
                    if j >= len(time_ranges):
                        break
                if j >= len(time_ranges):
                    break
                value_list[j] += 1
        for i in range(1, len(value_list)):
            value_list[i] += value_list[i - 1]
        time_ranges_ = []
        for i in time_ranges:
            time_ranges_.append(i[1])
        line = (
            pyecharts.charts.Line()
            .add_xaxis(time_ranges_)
            .add_yaxis('累计发言条数', value_list)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=ba_name + '吧-帖子' + str(i1 + 1) + "\n每15分钟发言数")
            )
            .render(path='Q13_' + str(i1 + 1) + '.html')
        )


def train():
    df = pretreat.wash_tie_table('孙笑川', 160)
    df_list = list(df['date'])
    from_time = df_list[0][:14] + '00'
    to_time = df_list[len(df_list) - 1][:14] + '00'
    frequency = 60 * 15
    time_ranges = split_time_ranges(from_time, to_time, frequency)
    value_list = [0] * len(time_ranges)
    j = 0
    for i in df_list:
        if (i <= time_ranges[j][1]) & (i >= time_ranges[j][0]):

            value_list[j] += 1
        else:
            while (i <= time_ranges[j][1]) & (i >= time_ranges[j][0]) != 1:
                j += 1
                if j >= len(time_ranges):
                    break
            if j >= len(time_ranges):
                break
            value_list[j] += 1
    for i in range(1, len(value_list)):
        value_list[i] += value_list[i - 1]
    time_ranges_ = []
    for i in time_ranges:
        time_ranges_.append(i[1])
    x = np.array(range(0, len(time_ranges_)))
    y = np.array(value_list)
    print('全体数据量：', len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    print('训练集数据量：', len(x_train))
    print('测试集数据量：', len(x_test))
    model = LinearRegression()
    in_x = x_train.reshape((len(x_train), 1))
    model.fit(in_x, y_train)
    print(model.coef_)
    pre_y = model.coef_ * x_train + model.intercept_

    plt.scatter(x, y)
    plt.plot(x_train, pre_y, color='r', label='y=%.2fx+%.2f' % (model.coef_, model.intercept_))
    plt.legend()
    plt.show()


def node_table():
    node_list = []
    file_list = os.listdir('./ba_info')
    for file in file_list:
        with open('./ba_info/' + file, 'r', encoding='utf-8') as fp:
            r_list = fp.read().split('\n')
        for info in r_list:
            info_list = info.split(' ')
            if len(info_list) >= 3:
                if int(info_list[2]) >= 10000000:
                    node = {
                        'name': info_list[0],
                        'symbolSize': int(info_list[2]) // 10000000,
                        'category': ba_index(info_list[4])
                    }
                    node_list.append(node)
    return node_list


def edge_table(node_name_list):
    links = []
    path_list = ['user_tables/剑网3/', 'user_tables/世界杯/', 'user_tables/孙笑川/', 'user_tables/bilibili/']
    for i in range(0, len(path_list)):
        path = path_list[i]
        file_list = os.listdir(path)
        for file in file_list:
            with open(path + file, 'r', encoding='utf-8') as fp:
                r_list = fp.read().split('\n')
            if len(r_list) < 3:
                continue
            ba_list = r_list[2].split(',')[:-1]
            for ba in ba_list:
                if ba not in node_name_list:


                    ba_list.remove(ba)
            for x in range(1, len(ba_list)):
                source_node_name = ba_list[0]
                target_node_name = ba_list[x]
                # if ba!=ba_name_list[i]:
                #     source_node_name =ba_name_list[i]
                #     target_node_name = ba
                # dict1[ba]=str(i)
                if {"source": source_node_name, "target": target_node_name} not in links:
                    links.append({"source": source_node_name, "target": target_node_name})
    return links


def ba_index(catagory):
    l1 = ['娱乐明星话题',
          '导演',
          '时尚人物',
          '明星',
          '粉丝组织',
          '网络红人',
          '选秀选手',
          'CP']
    l2 = ['DIY',
          '摄影',
          '画画',
          '旅行',
          '奢侈品',
          '手绘',
          '意境',
          '古玩',
          '模型',
          '彩票',
          '茶',
          '车',
          '创作',
          '二手',
          '交流',
          '经验',
          '花',
          '植物',
          '家居',
          '小而美',
          '理财',
          '投资',
          '职场',
          '多肉植物',
          '手工',
          '冷门收藏',
          '甜品',
          '美食',
          '购物',
          '变美',
          '留学移民',
          '文玩']
    l3 = ['桌游与休闲游戏',
          '游戏主播及平台',
          '游戏交易及功能',
          '游戏角色',
          '电子竞技及选手',
          '网络游戏',
          '其他游戏及话题',
          '单机与主机游戏',
          '手机游戏']
    l4 = ['偶像明星',
          '韩饭',
          '娱乐八卦',
          '欧美明星',
          '港台明星',
          '内地明星',
          '球星',
          '篮球明星',
          '笑星',
          '乐队',
          '时尚明星',
          '主持人',
          '声优控',
          '帅大叔',
          '日饭']
    l5 = ['CBA',
          'NBA',
          '国内足球',
          '国际足球',
          '篮球运动员',
          '综合体育',
          '综合运动员',
          '足球运动员',
          '运动教学',
          '运动装备']
    l6 = ['交通工具',
          '其他生活话题',
          '生活用品',
          '职业交流']
    l7 = ['公益慈善',
          '社会事件及话题',
          '社会机构']
    l8 = ['作家'
          '古典文学'
          '奇幻·玄幻小说'
          '文学期刊'
          '文学话题'
          '科幻文学'
          '诗词歌赋'
          '都市·言情小说']
    l9 = ['萌宠',
          '喵星人',
          '萝莉',
          '童年',
          '汪星人',
          '正太',
          '爆料',
          '吐槽',
          '内涵',
          '恐怖',
          '重口味',
          '星座']
    l10 = ['推理',
           '声优',
           'COS',
           '日本动漫',
           '国产动漫',
           '欧美动漫',
           '少男漫',
           '暴走漫画',
           '耽美漫画',
           '少女漫画',
           '搞笑漫画',
           '科幻系',
           '冒险',
           '青春恋爱',
           '热血动漫',
           '同人',
           '手办']
    l11 = ['日韩流行音乐',
           '欧美流行音乐',
           '民谣',
           '港台流行音乐',
           '电子',
           '韩国流行音乐',
           '音乐人及乐队',
           '音乐推荐',
           '音乐话题',
           '乐器乐理',
           '华语流行音乐',
           '古典',
           '嘻哈',
           '摇滚']
    l12 = ['国内地区',
           '海外地区']
    l13 = ['两性关系',
           '人际关系',
           '其他情感话题',
           '家庭关系']
    index_list = ['娱乐明星',
                  '生活家',
                  '游戏',
                  '追星族',
                  '体育',
                  '生活',
                  '文学',
                  '闲·趣',
                  '社会',
                  '动漫宅',
                  '音乐',
                  '地区',
                  '情感',
                  '次元文化',
                  '其他']
    l_list = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13]
    index = '其他'
    for l in l_list:
        if catagory in l:
            index = index_list[l_list.index(l)]
    if catagory == '次元文化':
        index = '次元文化'
    return index


def Q15():
    nodes = node_table()
    node_name_list=[]
    for i in nodes:
        node_name_list.append(i['name'])
    links = edge_table(node_name_list)



    g = (
        pyecharts.charts.Graph(init_opts=opts.InitOpts(width="1200px", height="900px"))
        .add("系列节点", nodes, links, repulsion=8000, is_draggable=True, is_roam=True,
             is_focusnode=True,
             # layout='circular',
             # categories=category,
             label_opts=opts.LabelOpts(is_show=True),
             linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.3, opacity=0.7))
        .set_global_opts(title_opts=opts.TitleOpts(title="吧之间的关系网络图"))
        .render('Q15.html')
    )


import xlwt
import xlsxwriter


def Q16(ba_name):
    workbook = xlsxwriter.Workbook('data.xlsx')
    sheet = workbook.add_worksheet('ba_content')

    sheet.write(0, 0, 'content')
    txt_list = os.listdir('tables/' + ba_name)
    for i in range(0, len(txt_list)):
        df = pretreat.wash_tie_table(ba_name, i)
        content_list = list(df['content'].astype(str))
        content = '，'.join(content_list)
        sheet.write(i, 0, content)
    workbook.close()


if __name__ == '__main__':
    ba_list = ['孙笑川', '世界杯', '剑网3', 'bilibili']
    # ba_list = ['孙笑川', '王者荣耀']
    # node_table()
    Q15()
    # score_list=getScore('孙笑川')
    # content_list=["买这个的不如把自己脑子抠出来挂身上，病毒看见你这芝麻大的脑子都不想感染你"
    #     ,'没事这样的人病毒都懒得感染怕拉低了病毒的智商'
    #     ,'少量的弹幕确实有意思，但是密密麻麻无关紧要的弹幕属实有点恶心人，影响观看了'
    #     ,'下水道的东西就老老实实待在下面，非要跑出来招摇逛市，恶心一堆人还不让别人说，怕评价就别发出来，我的评价是，恶心，非常恶心'
    #     ,'我就觉得长得好看的美，然后你跟我说美的定义很多种，就开始说我也又丑又肥'
    #     ,'南瓜也挺不错的其实，还有水煮蛋，水煮蛋，这人是真的能处，有事都真上，恰饭一般也都是掐自己的书'
    #     ,'那段红色经典讲解的很好，大决战系列看了几遍，配合弹幕还有下面的改编评论让整个视频更加有趣']
    # testemo(content_list)

    # pie = pyecharts.charts.Pie(init_opts=opts.InitOpts(theme='light',
    #                                   width='1000px',
    #                                   height='800px'))
    # pie.add("",
    #         [list(z) for z in zip(Faker.choose(), Faker.values())],
    #         radius=["20%", "50%"],
    #         center=["25%", "50%"])
    # # 添加多个饼图
    # pie.add("",
    #         [list(z) for z in zip(Faker.choose(), Faker.values())],
    #         radius=["20%", "50%"],
    #         center=["75%", "50%"])
    # print(Faker.values())
    # pie.render(path='Q4_1.html')
