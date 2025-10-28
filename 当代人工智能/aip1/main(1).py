import math
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import FastText
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from multiprocessing import Pool
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def readdata(txt_path):
    result = []
    with open(txt_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        tmp = {}
        for line in lines:
            t = eval(line[:-1])
            tmp['label'] = t['label']
            text = t['raw']
            text = text.lower()
            text = ''.join([char for char in text if char not in string.punctuation])
            words = text.split()
            words = [w for w in words if w not in stoplist]
            tmp['freq'] = Counter(words)
        result.append(tmp)
        print(result)
        exit(0)

def document_vector(model, doc):
    # 计算文本的词向量平均值
    doc = [word for word in doc if word in model.wv.index_to_key]
    return np.array(np.mean(model.wv[doc], axis=0))

# 定义一个函数用于并行计算特征向量
def parallel_document_vector(args):
    model, doc = args
    return document_vector(model, doc)

def readdata1(txt_path):
    result = []
    labels=[]
    with open(txt_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            t = eval(line[:-1])
            text=t['raw']
            labels.append(t['label'])
            # 使用str.translate()方法去除标点符号
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
            text = text.lower()

            # 去除停用词
            words = text.split(' ')
            tmp=[]
            for word in words:
                if word not in stoplist:
                    tmp.append(word)
            text=' '.join(tmp)

            result.append(text)
        return result,np.array(labels)

def text2vec(texts,pattern):
    # TD-IDF
    if pattern == 1:
        # 创建TF-IDF向量化器
        vector = TfidfVectorizer()
        # 计算TF-IDF值
        tf_idf = vector.fit_transform(texts)
        feature_names = vector.get_feature_names()
        return tf_idf.toarray(),feature_names

    # Word2Vec
    elif pattern == 2:
        sentences=[]
        for text in texts:
            sentences.append(text.split(' '))
        # model = Word2Vec(sentences,vector_size=100,window=5,min_count=2,workers=4,epochs=2)           0.1 依托答辩
        # model = Word2Vec(sentences,vector_size=100,window=5,min_count=1,workers=4)                    0.14 答辩
        # model = Word2Vec(sentences,vector_size=50,window=5,min_count=1,workers=4,epochs=10)             0.23
        model = Word2Vec(sentences,vector_size=30,window=5,min_count=1,workers=4,epochs=5)

        # 获取模型中的所有单词列表
        words = list(model.wv.index_to_key)

        # vectors =  [document_vector(model, doc) for doc in texts]
        # 创建一个进程池
        pool = Pool(processes=4)  # 4个进程

        # 使用进程池进行并行计算
        vectors = pool.map(parallel_document_vector, [(model, doc) for doc in texts])
        # 关闭进程池
        pool.close()
        pool.join()

        return vectors,words

    # Bert
    elif pattern == 3:
        # 加载预训练模型的分词器
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
        # 加载预训练模型并移至GPU
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        # model = AutoModelForMaskedLM.from_pretrained("./bert-base-uncased")

        results=np.array([],dtype=np.float32)
        all_words=[]

        # 对文本进行编码(可以批处理)
        # # 1
        count=0
        for text in texts:
            tokens = tokenizer.tokenize(text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

            # 将tokens转为torch tensors并移至GPU
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            count+=1
            print(count)
            # 获取隐藏状态
            with torch.no_grad():
                outputs = model(tokens_tensor)
                hidden_states = outputs.last_hidden_state
            # np.concatenate((results,hidden_states.cpu().numpy()))
            results=np.vstack((results,hidden_states.cpu().numpy().squeeze()))
            all_words.extend(tokens)


        # batch_size=8
        # for i in range(0,len(texts),batch_size):
        #     batch_texts=texts[i:i+batch_size]
        #     tokens=tokenizer.batch_encode_plus(batch_texts,padding=True,truncation=True,return_tensors="pt")
        #     tokens_tensor = tokens["input_ids"].to(device)
        #     with torch.no_grad():
        #         outputs = model(tokens_tensor)
        #         hidden_states = outputs.last_hidden_state
        #     results.append(hidden_states.cpu().numpy())
        #     all_words.extend(tokens["input_ids"].cpu().numpy())


        return results,all_words

    # FastText
    elif pattern == 4:
        sentences=[]
        for text in texts:
            sentences.append(text.split(' '))
        model = FastText(sentences,vector_size=100,window=5,min_count=1,workers=4)
        # 获取模型中的所有单词列表
        words = list(model.wv.index_to_key)
        # 获取每个单词对应的词向量
        vectors = [model.wv[word] for word in words]
        # 将词向量列表转换为 NumPy 数组
        vectors_matrix = np.array(vectors)
        return vectors_matrix,model.wv.index_to_key

def train(X,y,pattern):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    if pattern==1:
        # 初始化逻辑回归模型，使用一对多策略
        clf = LogisticRegression(random_state=42,multi_class='ovr')
    elif pattern==2:
        # 决策树
        clf = DecisionTreeClassifier(random_state=42)
    elif pattern==3:
        # 支持向量机
        clf = SVC(kernel='linear',random_state=42)
    clf.fit(X_train,y_train)
    # 预测
    y_pred=clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print('accuracy:',accuracy)

def train_MLP(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 设置超参数
    input_dim = X_train.shape[1]
    hidden_dim = 100
    output_dim = len(set(y_train))
    learning_rate = 0.01
    epochs = 10

    # 实例化模型、损失、优化器
    model = MLP(input_dim,hidden_dim,output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    # 训练模型
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        # outputs = outputs.detach()
        # outputs=torch.tensor(outputs)
        loss = criterion(outputs,y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    # 预测模型
    with torch.no_grad():
        predictions = model(X_test_tensor)
        _, predicted = torch.max(predictions,1)
        accuracy =(predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f'Test Accuracy: {accuracy*100:.2f}%')

# TF-IDFtransform有一点用，但没什么大用
if __name__ == '__main__':
    # 读取数据集
    documents,y=readdata1('../exp1_data/train_data.txt')
    # print(documents)


    # 将文本映射成向量
    X, feature_names=text2vec(documents,2)
    print(X)
    print(len(X),len(X[0]))
    # print(feature_names)

    train(X,y,1)
    # train_MLP(X,y)