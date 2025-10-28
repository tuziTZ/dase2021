import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pycocoevalcap.cider.cider import Cider
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
# from nltk.translate import rouge_score
from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import sentence_ngram_bleu
# from nltk.translate import ciderscorer
from rouge import Rouge
import numpy as np
from tqdm import tqdm

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 读取CSV文件
csv_path = "train.csv"  # 替换为你的CSV文件路径
df = pd.read_csv(csv_path)

# 提取描述和诊断
descriptions = df["description"].tolist()
diagnoses = df["diagnosis"].tolist()


# 将描述转换为数字序列
def text_to_sequence(text):
    return [int(token) for token in text.split()]


# 转换为数字序列
sequences = [text_to_sequence(desc) for desc in descriptions]
diagnoses = [text_to_sequence(desc) for desc in diagnoses]

# 划分训练集和测试集
train_sequences, val_sequences, train_diagnoses, val_diagnoses = train_test_split(
    sequences, diagnoses, test_size=0.2, random_state=42
)
all_in=train_sequences + val_sequences
all_out=train_diagnoses + val_diagnoses
print(max(len(sequence) for sequence in all_in))
print(max(len(sequence) for sequence in all_out))


# 将训练集和测试集合并
all_sequences = train_sequences + val_sequences + train_diagnoses + val_diagnoses


# 将整数序列展平为单个列表
all_words = [word for sequence in all_sequences for word in sequence]

# 使用集合来获取唯一的单词
vocab_set = set(all_words)

# 获取词表大小
vocab_size = len(vocab_set)
