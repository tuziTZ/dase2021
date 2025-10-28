import random


def readData():
    with open('train_data.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
    data_list = []
    for line in data:
        data_list.append(eval(line))
    return data_list


def subset(alist, index_list):
    result = []
    for i in index_list:
        result.append(alist[i])
    return result


def split_list(alist, group_num=10, shuffle=True):
    index = list(range(len(alist)))
    if shuffle:
        random.shuffle(index)

    elem_num = len(alist) // group_num
    sub_lists = {}
    for idx in range(group_num):
        start, end = idx * elem_num, (idx + 1) * elem_num
        sub_lists['set' + str(idx)] = subset(alist, index[start:end])

    return sub_lists


# 读取老师给出的数据
data_list = readData()
# 将训练集划分为10个子集，交叉训练
result = split_list(data_list, group_num=10, shuffle=True)
# print(result['set0'])


from sklearn.feature_extraction.text import TfidfVectorizer

def sklearn_tfidf(alist):
    transformer = TfidfVectorizer()
    tfidf = transformer.fit_transform(alist)
    return tfidf.toarray()


# 选定特定的验证集，得到text_list和tag_list
text_list_train = []
tag_list_train = []
text_list_val = []
tag_list_val = []
val=8
for idx in range(10):
    if idx==val:
        for d in result['set' + str(idx)]:
            text_list_val.append(d['raw'])
            tag_list_val.append(d['label'])
    else:
        for d in result['set' + str(idx)]:
            text_list_train.append(d['raw'])
            tag_list_train.append(d['label'])
print(len(text_list_val),len(text_list_train))
text_list_all=text_list_train+text_list_val
# 将文本转化为向量
x_all = sklearn_tfidf(text_list_all)
x_train=x_all[:7200]
x_val=x_all[7200:]

# 逻辑回归logical regression
from sklearn.linear_model import LogisticRegression
lrmodel=LogisticRegression()
lrmodel.fit(x_train,tag_list_train)
print(lrmodel.score(x_val,tag_list_val))

# 支持向量机svm
from sklearn import svm
svmmodel=svm.SVC(kernel='linear')
svmmodel.fit(x_train,tag_list_train)
print(svmmodel.score(x_val,tag_list_val))


