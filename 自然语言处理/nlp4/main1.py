import math
import os
import time
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM


# 定义跳表节点类
class SkipListNode:
    def __init__(self, name, value):
        self.value = value
        self.name = name
        self.next = None
        self.down = None


# 定义跳表类
class SkipList:
    def __init__(self):
        self.head = SkipListNode('', float('-inf'))
        self.tail = SkipListNode('', float('inf'))
        self.head.next = self.tail
        self.size = 0

    # 插入节点
    def insert(self, name, value):
        node = SkipListNode(name, value)
        current = self.head
        while current:
            if current.next.value > value:
                break
            else:
                current = current.next
        node.next = current.next
        current.next = node
        self.size += 1

    # 构建上层节点
    def populate_up_level(self):
        current = self.head
        up_current = None
        level = 0
        while current:
            if level % math.isqrt(self.size) == 0:
                up_node = SkipListNode('up', current.value)
                up_node.down = current
                current.down = up_node
                if up_current:
                    up_current.next = up_node
                up_current = up_node
            if current.next:
                current = current.next
            level += 1

    # 利用上层节点，查询值为value的node是否存在在跳表中，返回True或False
    def search(self, value):
        current = self.head
        while current:
            if current.value == value:
                return True
            # 继续向前查询
            elif current.value < value:
                # 进入跳表查询
                if current.down:
                    if current.down.next.value <= value:
                        current = current.down.next.down
                # 无跳跃指针
                else:
                    current = current.next
        return False

    # 返回一个列表，包含跳表中所有节点的(name,value)元组
    def return_list(self):
        result = []
        current = self.head
        while current:
            if current.next:
                result.append((current.next.name, current.next.value))
                current = current.next
            else:
                if current.down:
                    current = current.down
                else:
                    break
        return result


# 加载停用词
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return stopwords


# 加载文档
def load_documents(directory):
    documents = []
    documents_name = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read())
            documents_name.append(filename)
    return documents, documents_name


# 分词
# def tokenize(text):
#     tokens = jieba.lcut(text)
#     return [token for token in tokens if token not in stopwords]
# 读取词汇文件
with open("corpus.dict.txt", "r", encoding="utf-8") as f:
    dictionary = set(line.strip() for line in f)


def max_match_segment(text):
    # 从文本中匹配最长的词汇
    seg_list = []
    text_len = len(text)
    while text_len > 0:
        max_len = min(7, text_len)  # 最大长度限制为12个字符
        for i in range(max_len, 0, -1):
            word = text[:i]
            if word in dictionary:
                seg_list.append(word)
                text = text[i:]
                text_len = len(text)
                break
        else:
            seg_list.append(text[0])
            text = text[1:]
            text_len = len(text)
    return seg_list


def tokenize(text):
    tokens = max_match_segment(text)
    return [token for token in tokens if token not in stopwords]


# 建立倒排索引和词向量字典
def build_inverted_index_and_word_vectors(documents, documents_name, tokenizer, model):
    inverted_index = {}
    word_vectors = {}
    for doc_id, document in enumerate(documents):
        tokens = tokenize(document)
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = SkipList()
                word_vectors[token] = encode_word(token, tokenizer, model)
            inverted_index[token].insert(documents_name[doc_id], doc_id)

    return inverted_index, word_vectors


# 使用BERT模型编码词语
def encode_word(word, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(word, add_special_tokens=True)])
    with torch.no_grad():
        outputs = model(input_ids)

    last_hidden_states = outputs.logits
    return last_hidden_states.mean(dim=1).squeeze().numpy()


# 计算相似度
def calculate_similarity(word, word_vectors):
    try:
        vec1 = word_vectors[word]
    except KeyError:
        print("词表中不存在词汇：" + word)
        return None

    similarities = {}
    for w, vec2 in word_vectors.items():
        similarities[w] = cosine_similarity([vec1], [vec2])[0][0]
    # 按相似度从高到低排序
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    # 返回与指定词语义相似度最高的Top2个词
    return sorted_similarities[1:3]


# 主函数
def process_text_similarity(documents_directory, stopwords_file):
    # 加载停用词
    global stopwords
    stopwords = load_stopwords(stopwords_file)

    # 加载文档
    documents, documents_name = load_documents(documents_directory)
    # 加载BERT模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")
    print("建立倒排索引，保存词向量...")
    inverted_index, word_vectors = build_inverted_index_and_word_vectors(documents, documents_name, tokenizer, model)
    print("给每个倒排索引构建跳表...")
    for key in inverted_index.keys():
        inverted_index[key].populate_up_level()
    while True:
        # 输入指定词
        target_word = input("请输入要查询的词语（输入exit退出）：")
        if target_word.lower() == "exit":
            break

        start_time = time.time()

        # 计算相似词
        similar_words = calculate_similarity(target_word, word_vectors)
        if similar_words is None:
            continue

        # 找到与指定词语义相似度最高的Top2个词同时包含的文档ID
        relevant_doc_ids = inverted_index[target_word].return_list()
        for word, _ in similar_words:
            relevant_doc_ids1 = inverted_index[word]  # 一个跳表
            for name, value in relevant_doc_ids:
                if relevant_doc_ids1.search(value) is False:
                    relevant_doc_ids.remove((name, value))

        end_time = time.time()

        # 输出结果
        print("指定词:", target_word)
        print("与指定词语义相似度最高的Top2个词:", similar_words)
        print("最终输出的文档ID:", list(relevant_doc_ids))
        print("检索时间:", end_time - start_time, "秒")


# 示例用法
if __name__ == "__main__":
    process_text_similarity("./article", "cn_stopwords.txt")
