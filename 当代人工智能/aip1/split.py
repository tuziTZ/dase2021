import random
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
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


def transformers_Bert(alist):
    # 加载预训练模型的分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    # 加载预训练模型并移至GPU
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    # model = AutoModelForMaskedLM.from_pretrained("./bert-base-uncased")

    results = np.array([], dtype=np.float32)
    all_words = []

    count = 0
    for a in alist:
        tokens = tokenizer.tokenize(a)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

        # 将tokens转为torch tensors并移至GPU
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        count += 1

        # 获取隐藏状态
        with torch.no_grad():
            outputs = model(tokens_tensor)
            hidden_states = outputs.last_hidden_state
        # np.concatenate((results,hidden_states.cpu().numpy()))
        if count==1:
            results = hidden_states.cpu().numpy().squeeze()
        else:
            results = np.vstack((results, hidden_states.cpu().numpy().squeeze()))

        all_words.extend(tokens)
    return results, all_words


# 选定特定的验证集，得到text_list和tag_list
text_list_train = []
tag_list_train = []
text_list_val = []
tag_list_val = []
val = 8
for idx in range(10):
    if idx == val:
        for d in result['set' + str(idx)]:
            text_list_val.append(d['raw'])
            tag_list_val.append(d['label'])
    else:
        for d in result['set' + str(idx)]:
            text_list_train.append(d['raw'])
            tag_list_train.append(d['label'])
print(len(text_list_val), len(text_list_train))
text_list_all = text_list_train + text_list_val
# 将文本转化为向量
# print(text_list_all[0])
# x_all = sklearn_tfidf(text_list_all)
x_all = transformers_Bert(text_list_all)

x_train = x_all[:7200]
x_val = x_all[7200:]


# 逻辑回归logical regression
from sklearn.linear_model import LogisticRegression

lrmodel = LogisticRegression(multi_class='ovr')
lrmodel.fit(x_train, tag_list_train)
# print(lrmodel.score(x_val,tag_list_val))
tag_list_pred = lrmodel.predict(x_val)
accuracy = accuracy_score(tag_list_val, tag_list_pred)
print(f"Accuracy:{accuracy:.2f}")


# 支持向量机svm
from sklearn import svm

svmmodel = svm.SVC(kernel='linear', random_state=42)
svmmodel.fit(x_train, tag_list_train)
print(svmmodel.score(x_val, tag_list_val))
