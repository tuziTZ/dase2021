import torch


def readData():
    with open('train_data.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
    data_list = []
    for line in data:
        data_list.append(eval(line))
    return data_list


data_list = readData()
text_list = []
tag_list = []
for d in data_list:
    text_list.append(d['raw'])
    tag_list.append(d['label'])
print(len(text_list), len(tag_list))


def readTest():
    test_list = []
    with open('test.txt', 'r', encoding='utf-8') as fp:
        # 跳过标题行（如果有的话）
        next(fp)
        for line in fp:
            # 移除换行符并分割成id和text两部分
            line = line.strip()
            id, text = line.split(', ', 1)
            test_list.append(text)
    return test_list


test_list = readTest()
text_list_all = text_list + test_list

import string
import re


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()

    preprocessed_text = ' '.join(words)

    return preprocessed_text


# print(text_list_all[0])
text_list_all = [preprocess_text(sentence) for sentence in text_list_all]

from sklearn.feature_extraction.text import TfidfVectorizer


def sklearn_tfidf(alist):
    transformer = TfidfVectorizer(stop_words='english',min_df=2,max_df=0.9)
    # transformer = TfidfVectorizer()
    tfidf = transformer.fit_transform(alist)
    return tfidf.toarray()


x_all = sklearn_tfidf(text_list_all)
x_train = x_all[:8000]
x_test = x_all[8000:]

import torch.nn as nn
import torch.optim as optim

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(tag_list, dtype=torch.int64)
x_val_tensor = torch.tensor(x_test, dtype=torch.float32)


# y_val_tensor = torch.tensor(tag_list_val, dtype=torch.int64)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 30)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(30, output_dim)
        #         self.relu = nn.ReLU()
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        #         x = self.relu(self.fc3(x))
        #         x = self.dropout(x)
        x = self.fc4(x)
        return x


input_dim = len(x_train[0])
hidden_dim = 100
output_dim = len(set(tag_list))
learning_rate = 0.01
epochs = 25

print(input_dim, hidden_dim, output_dim)

mlp = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
# optimizer=optim.Adam(mlp.parameters(), lr=learning_rate)
optimizer = optim.RMSprop(mlp.parameters(), lr=learning_rate)

# 对于每种参数组合，保存每个epoch时模型在验证集上的accuracy在列表中
# accuracy_list=[]

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = mlp(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch{epoch + 1}/{epochs},Loss:{loss.item():.4f}")
# with torch.no_grad():  # 去掉这一行就会变得稳定在96%
predictions = mlp(x_val_tensor)
_, predicted = torch.max(predictions, 1)
print(list(predicted))
# accuracy = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
# print(f"Test Accuracy:{accuracy * 100:.2f}%")
#     accuracy_list.append(round(accuracy,4))
# print(accuracy_list)


pred_data = list(predicted)

# 指定要写入的文本文件路径
output_file = "output.txt"

# 打开文件并将数据写入
with open(output_file, 'w') as file:
    # 写入标题行
    file.write("id, pred\n")

    # 写入数据行
    for i, pred in enumerate(pred_data):
        file.write(f"{i}, {pred}\n")
