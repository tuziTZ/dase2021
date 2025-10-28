#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 数据源结构
# --tables--[吧名]--0.txt
#         |      | 1.txt
#         |      --....txt
#         |...
#         --[吧名]

# --user_tables--[吧名]--0.txt
#             |      | 1.txt
#             |      --....txt
#             |...
#             --[吧名]
#
# 数据量:  bilibili:2119
#         csgo:50
#         第五人格交易:511
#         剑网3:555       1723users
#         艾欧尼亚:509      主题帖
#         抗压背锅:49
#         世界杯:47
#         王者荣耀:55
#         孙笑川:318       20021users      主题帖
#         图拉丁:86

import os
import traceback
import pandas as pd
import json
import re
import time
import jieba
import jieba.analyse


# 对一个帖的数据进行预处理
def wash_tie_table(ba_name, num):
    # 将发帖内容的','替换为'，'
    with open('tables/' + ba_name + '/' + str(num) + '.txt', 'r', encoding='utf-8') as fp:
        text = fp.read()
    tmp = text.split('\n')
    new_text = []
    for line in tmp[2:-1]:
        split_line = line.split(',')
        context = ''.join(split_line[5:-3])
        context = context.replace(',', '，')
        context_list = [context]

        new_line = split_line[:5] + context_list + split_line[-3:]
        new_text1 = ','.join(new_line)
        new_text.append(new_text1)
    new_text = tmp[1:2] + new_text + tmp[-1:]
    text = '\n'.join(new_text)
    # 创建csv文件，将可以建成数据表的数据写入
    csv_path = 'test.csv'

    with open(csv_path, 'w', encoding='utf-8') as fp:
        fp.write(text)
    # df是目标文件（代表一个帖子）建成的数据表
    df = pd.DataFrame(pd.read_csv(csv_path, header=0))

    # 对数据表进行预处理
    df['is_lz'].fillna(value=0, inplace=True)
    df.drop_duplicates(subset='floor', inplace=True)
    df.dropna(inplace=True)

    df.reset_index(inplace=True)
    return df


# 合并同一个吧中帖子的数据表
def combine(ba_name):
    txt_list = os.listdir('tables/' + ba_name)
    obj_list = []

    for i in range(0, len(txt_list)):
        obj_list.append(wash_tie_table(ba_name, i))
    df = pd.concat(obj_list)
    # df.info()
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)

    return df


# 对一个文件夹中的用户数据(三项)进行预处理
def wash_user_table(ba_name):
    txt_list = os.listdir('user_tables/' + ba_name)
    obj_list = []
    for i in txt_list:
        with open('user_tables/' + ba_name + '/' + i, 'r', encoding='utf-8') as fp:
            line = fp.readline()
        sex = ''
        if '男' in line:
            sex = '男'
        elif '女' in line:
            sex = '女'
        age = ''
        age_list = re.findall('吧龄:(.*?)年', line)
        if len(age_list) == 0:
            age = ''
        else:
            age = age_list[0]
        num = ''
        num_list = re.findall('发贴:(.*?),', line)
        if len(num_list) == 0:
            num = ''
        else:
            num = num_list[0]
        if num[-1:] == '万':
            num = str(int(float(num[:-1]) * 10000))
        tmp_obj = [sex, age, num]

        obj = ','.join(tmp_obj)
        obj_list.append(obj)
    r = 'sex,age,num' + '\n' + '\n'.join(obj_list)
    with open('test.csv', 'w', encoding='utf-8') as fp:
        fp.write(r)
    df = pd.DataFrame(pd.read_csv('test.csv', header=0))
    return df


def wash_user_table_(ba_name):
    txt_list = os.listdir('user_tables/' + ba_name)
    obj_list = []
    for i in txt_list:
        with open('user_tables/' + ba_name + '/' + i, 'r', encoding='utf-8') as fp:
            r = fp.read()
        r_list = r.split('\n')
        if len(r_list) < 3:
            continue
        line = r_list[0]
        sex = ''  # 性别用0,1来表示
        if '男' in line:
            sex = '0'
        elif '女' in line:
            sex = '1'
        age = ''
        age_list = re.findall('吧龄:(.*?)年', line)
        if len(age_list) == 0:
            age = ''
        else:
            age = age_list[0]
        num = ''
        num_list = re.findall('发贴:(.*?),', line)
        if len(num_list) == 0:
            num = ''
        else:
            num = num_list[0]
        if num[-1:] == '万':
            num = str(int(float(num[:-1]) * 10000))

        follow_list = r_list[1].split(',')[:-1]
        follow_peo = len(follow_list)
        follow_ba = len(r_list[2].split(',')[:-1])
        user_href = i[:-4]
        tmp_obj = [user_href, sex, age, num, str(follow_peo), str(follow_ba)]

        obj = ','.join(tmp_obj)
        obj_list.append(obj)

    r = 'user_href,sex,age,num,follow_peo,follow_ba' + '\n' + '\n'.join(obj_list)
    with open('test.csv', 'w', encoding='utf-8') as fp:
        fp.write(r)
    df = pd.DataFrame(pd.read_csv('test.csv', header=0))
    df.dropna(inplace=True)
    return df


# 根据需要，对数据表进行进一步处理


# 1.从数据表中提取用户名和ip地址
def ba_ip(df):
    # 删除不需要的列
    df.drop(labels=['id', 'user_level', 'is_lz', 'date', 'floor', 'content'], axis=1, inplace=True)
    # 确保没有重复用户
    df.drop_duplicates(subset='user_href', inplace=True)
    df = df.sample(n=30000, replace=False)
    value = []
    attr = ['甘肃', '广东', '广西', '贵州', '海南',
            '河南', '湖北', '湖南', '宁夏', '青海',
            '陕西', '四川', '西藏', '新疆', '云南',
            '重庆', '北京', '天津', '河北', '山西', '内蒙古',
            '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建'
        , '江西', '山东']
    ip_sum = df['ip_address'].count()
    for pro in attr:
        value.append((int(df.loc[df['ip_address'] == 'IP属地:' + pro].ip_address.count())))
    sequence = list(zip(attr, value))
    return sequence


# 2.统计相同ip地址的用户平均发言频数
def ip_fre(df):
    df.drop(labels=['id', 'user_level', 'is_lz', 'date', 'floor', 'content'], axis=1, inplace=True)
    attr = ['甘肃', '广东', '广西', '贵州', '海南',
            '河南', '湖北', '湖南', '宁夏', '青海',
            '陕西', '四川', '西藏', '新疆', '云南',
            '重庆', '北京', '天津', '河北', '山西', '内蒙古',
            '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建'
        , '江西', '山东']
    value = []
    for pro in attr:
        df1 = df.loc[df['ip_address'] == 'IP属地:' + pro]
        cnt = df1['user_href'].value_counts().mean()
        if cnt:
            value.append(cnt)
        else:
            value.append(0)
    sequence = list(zip(attr, value))
    return sequence


# 3.统计一个吧的平均删楼率
def tie_delete(ba_name):
    txt_list = os.listdir('tables/' + ba_name)
    tie_delete_fre = []
    for i in range(0, len(txt_list)):
        df = wash_tie_table(ba_name, i)

        df.drop(labels=['id', 'user_level', 'is_lz', 'date', 'ip_address', 'content', 'user_href', 'user_name'], axis=1,
                inplace=True)
        df.dropna(inplace=True)
        floor_sum = df['floor'].count()
        if len(df['floor']) >= 2:
            floor_max = int(df['floor'].loc[floor_sum - 1][:-1])
        else:
            continue

        tie_delete_fre.append((floor_max - floor_sum) / floor_max)

    return tie_delete_fre


# 4.性别比例与吧龄分布
def sex(df):
    df.drop(labels='num', axis=1, inplace=True)

    sex_list = ['男', '女']
    num_list = []
    for i in sex_list:
        num_list.append(int(df.loc[df['sex'] == i].sex.count()))

    return sex_list, num_list


def age(df):
    df['age'].dropna()
    bins = range(0, 15)
    segments = pd.cut(x=df["age"], bins=bins, right=True)
    counts = pd.value_counts(segments, sort=False)
    attr = list(counts.index.astype(str))
    value = list(counts)
    return attr, value


# 5.楼主发言在帖子中的占比
def lz(df):
    df.dropna()
    num_list = []
    for i in range(0, 2):
        num_list.append(df.loc[df['is_lz'] == i].is_lz.count())
    if sum(num_list) != 0:

        return num_list[1] / sum(num_list)
    else:
        return 0


def level(df):
    counts = df['user_href'].value_counts(sort=False)
    df = df.drop_duplicates(subset='user_href')
    df.insert(loc=2, column='counts', value=list(counts))
    df['user_level'] = df['user_level'].astype(int)
    df.sort_values('user_level', inplace=True)
    df_level = df['user_level']
    action_list = []
    df.sort_values('counts', inplace=True)
    # print(df)
    # df=df.head(-4)
    # print(df)
    # bins = range(0, 15)
    # segments = pd.cut(x=list(df), bins=bins, right=True)
    attr = list(pd.unique(df_level))
    for i in attr:
        new_df = df.loc[(df['user_level'] == i), ['counts']].sum() / df.loc[(df['user_level'] == i), ['counts']].count()
        action_list.append(float(new_df))

    counts = df_level.value_counts(sort=False)

    value = list(counts)

    return attr, value, action_list


# def date(df):
#     df.drop(labels=['id', 'user_level', 'is_lz', 'date', 'ip_address', 'content', 'user_href', 'user_name'], axis=1,
#             inplace=True)
def time_(df):
    df.drop(labels=['id', 'level_0', 'index', 'floor', 'user_level', 'is_lz', 'ip_address', 'content', 'user_href',
                    'user_name'], axis=1,
            inplace=True)
    df.dropna()
    df = df.sample(n=3000, replace=False)
    time_list = list(df['date'])
    time_list_ = []
    for i in time_list:
        time_list_.append(i[-5:])

    time_list_.sort()
    # 按小时划分
    time_table = []
    for i in range(0, 24):
        if i < 10:
            time_table.append('0' + str(i) + ':00')
        else:
            time_table.append(str(i) + ':00')

    time_index = []
    x = 0
    j = time_list_[x]
    for i in time_table:
        cnt = 0
        while (j < i):
            cnt += 1
            x += 1
            j = time_list_[x]
        time_index.append(cnt)
    time_index = time_index[1:]
    return time_index, time_table


def date(ba_name):
    csv_path = 'test.csv'
    txt_path = 'Zhutitietables/' + ba_name + '.txt'
    with open(txt_path, 'r', encoding='utf-8') as fp:
        r = fp.read()
    line_list = r.split('\n')
    line_list_ = []
    for line in line_list:
        num = line.count(',')
        if num > 1:
            line1 = line.replace(',', '，', num - 1)
        else:
            line1 = line
        line_list_.append(line1)

    with open(csv_path, 'w', encoding='utf-8') as fp:
        fp.write('title,date\n')
        fp.write('\n'.join(line_list_))

    df = pd.DataFrame(pd.read_csv(csv_path))
    df = df.drop_duplicates()

    date_list = list(df['date'])
    for i in range(0, len(date_list)):
        if ':' in date_list[i]:
            date_list[i] = '2022-12-22'  # 最近的日期
        else:
            date_list[i] = '2022-' + date_list[i]
    date_list.sort()

    df = pd.DataFrame(date_list)

    value = list(df.value_counts().astype('int'))
    attr = df.value_counts().index
    attr1 = []
    for i in range(0, len(attr)):
        attr1.append(list(attr[i])[0])

    data = list(zip(attr1, value))
    print(data)
    return data


def emo(df):
    df_new = df.loc[(df['date'] >= '2022-12-18 00:00')]
    df_old = df.loc[(df['date'] < '2022-12-18 00:00')]
    df_new.drop(labels=['id', 'level_0', 'index', 'floor', 'user_level', 'is_lz', 'ip_address', 'date', 'user_href'],
                axis=1,
                inplace=True)
    df_new['user_name'] = df_new['user_name'].astype(str)
    df_new['content'] = df_new['content'].astype(str)
    name_str = ''.join(df_new['user_name'])
    content_str = ''.join(df_new['content'])
    myre = re.compile(u'['
                      u'\U0001F300-\U0001F64F'
                      u'\U0001F680-\U0001F6FF'
                      u'\u2600-\u2B55]+',
                      re.UNICODE)
    all_emoji = ''.join(myre.findall(name_str) + myre.findall(content_str))
    emoji_table = {}
    for i in all_emoji:
        if i not in emoji_table:
            emoji_table[i] = 1
        else:
            emoji_table[i] += 1
    data_new = sorted(emoji_table.items(), key=lambda x: x[1])

    df_old.drop(labels=['id', 'level_0', 'index', 'floor', 'user_level', 'is_lz', 'ip_address', 'date', 'user_href'],
                axis=1,
                inplace=True)
    df_old['user_name'] = df_old['user_name'].astype(str)
    df_old['content'] = df_old['content'].astype(str)
    name_str1 = ''.join(df_old['user_name'])
    content_str1 = ''.join(df_old['content'])
    all_emoji1 = ''.join(myre.findall(name_str1) + myre.findall(content_str1))
    emoji_table1 = {}
    for i in all_emoji1:
        if i not in emoji_table1:
            emoji_table1[i] = 1
        else:
            emoji_table1[i] += 1

    data_old = sorted(emoji_table1.items(), key=lambda x: x[1])

    return data_new, data_old


def emo_(df):
    df.drop(labels=['id', 'level_0', 'index', 'floor', 'user_level', 'is_lz', 'ip_address', 'date', 'user_href'],
            axis=1,
            inplace=True)
    df['user_name'] = df['user_name'].astype(str)
    df['content'] = df['content'].astype(str)
    name_str = ''.join(df['user_name'])
    content_str = ''.join(df['content'])
    myre = re.compile(u'['

                      u'\U0001F300-\U0001F64F'

                      u'\U0001F680-\U0001F6FF'

                      u'\u2600-\u2B55]+',

                      re.UNICODE)
    all_emoji = ''.join(myre.findall(name_str) + myre.findall(content_str))
    emoji_table = {}
    for i in all_emoji:
        if i not in emoji_table:
            emoji_table[i] = 1
        else:
            emoji_table[i] += 1

    data = sorted(emoji_table.items(), key=lambda x: x[1])
    return data


def hot_words(ba_name):
    df = combine(ba_name)
    content_str = ','.join(list(df['content'].astype(str)))

    tags = jieba.analyse.extract_tags(content_str, topK=50, withWeight=True,
                                      allowPOS=("n", 'nr', 'ns', 'nt', 'nw', 'nz'))
    new_tags = []
    stop_list = ['楼主', '借楼', '帖子', '时候', '有点', '感觉', '东西']
    for i in tags:
        if i[0] not in stop_list:
            new_tags.append(i)
    return new_tags


def corr(ba_name):
    # 对用户的各项指标求相关矩阵，寻找相关性强的变量
    # 在A吧发言数,在A吧作为楼主发言条数,在A吧的等级,性别（0,1）,吧龄,ip地址（按照编号）,关注的人数,关注的吧数,总发帖数
    # counts,counts_lz,user_level,sex,age,ip_address,follow_peo,follow_ba,num
    ba_df = combine(ba_name)
    user_df = wash_user_table_(ba_name)
    ba_df.drop(labels=['level_0', 'index', 'id', 'user_name', 'content', 'date', 'floor'], axis=1, inplace=True)
    pro_attr = ['甘肃', '广东', '广西', '贵州', '海南',
                '河南', '湖北', '湖南', '宁夏', '青海',
                '陕西', '四川', '西藏', '新疆', '云南',
                '重庆', '北京', '天津', '河北', '山西', '内蒙古',
                '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建'
        , '江西', '山东']
    for i in pro_attr:
        ba_df['ip_address'].replace('IP属地:' + i, pro_attr.index(i), inplace=True)
    for i in ba_df['ip_address']:
        if i not in range(0, len(pro_attr)):
            ba_df['ip_address'].replace(i, '32', inplace=True)
    ba_df['is_lz'] = ba_df['is_lz'].astype(int)
    ba_df['ip_address'] = ba_df['ip_address'].astype(int)
    ba_df['user_level'] = ba_df['user_level'].astype(int)

    user_list = list(ba_df['user_href'].unique())
    lz_list = [0] * len(user_list)
    for i in range(0, len(ba_df['is_lz'])):
        if ba_df['is_lz'][i] == 1:
            index = user_list.index(ba_df['user_href'][i])
            lz_list[index] += 1

    counts = ba_df['user_href'].value_counts(sort=False)
    df = ba_df.drop_duplicates(subset='user_href')
    df.insert(loc=2, column='counts', value=list(counts))
    df.insert(loc=2, column='counts_lz', value=lz_list)
    df.drop(labels=['is_lz'], axis=1, inplace=True)
    for i in list(df['user_href']):
        x = i.split('.')[2]
        df['user_href'].replace(i, x, inplace=True)
    df['user_href'] = df['user_href'].sort_values()
    user_df['user_href'] = user_df['user_href'].sort_values()
    ba_df = df.set_index('user_href')
    user_df = user_df.set_index('user_href')

    new_df = pd.merge(ba_df, user_df, right_on='user_href', left_index=True, how='inner')
    return new_df
    # index_list=['']*len(df['user_href'])
    # for i in range(0,len(df['user_href'])):
    #     for j in range(0,len(user_df['user_href'])):
    #         user=list(user_df['user_href'])[j]
    #         if user in list(df['user_href'])[i]:
    #             index_list[i]=j
    #             break
    # print(index_list)


def line(ba_name):
    df = wash_user_table_(ba_name)
    x_data = list(df['age'].astype(float).sort_values())
    y_data = list(df['num'])
    return x_data, y_data


def pick_most_floor(ba_name):
    txt_list = os.listdir('tables/' + ba_name)
    obj_list = []
    # 选取楼数最多，删帖率最少的几个帖子进行统计
    for i in range(0, len(txt_list)):
        df = wash_tie_table(ba_name, i)
        if len(df) > 1000:
            floor_sum = df['floor'].count()
            floor_max = int(df['floor'].loc[floor_sum - 1][:-1])
            fre = float((floor_max - floor_sum) / floor_max)
            if fre < 0.2:
                obj_list.append(df['date'].sort_values())
                print(i)
    return obj_list




if __name__ == '__main__':
    # df = wash_user_table('孙笑川')
    pick_most_floor('孙笑川')
    # emo(df)
