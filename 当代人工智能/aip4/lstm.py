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

# 划分训练集和测试集
train_sequences, val_sequences, train_diagnoses, val_diagnoses = train_test_split(
    sequences, diagnoses, test_size=0.2, random_state=42
)


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, sequences, diagnoses):
        self.sequences = sequences
        self.diagnoses = diagnoses

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index])
        diagnosis = torch.tensor([int(token) for token in self.diagnoses[index].split()])
        return sequence, diagnosis, len(sequence)


# 创建数据加载器
batch_size = 32
train_dataset = CustomDataset(train_sequences, train_diagnoses)
val_dataset = CustomDataset(val_sequences, val_diagnoses)


# 填充序列函数
def pad_collate(batch):
    sequences, diagnoses, _ = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_diagnoses = pad_sequence(diagnoses, batch_first=True, padding_value=0)
    return padded_sequences, padded_diagnoses, _


# 重新创建数据加载器，使用pad_collate函数
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, input_seq, input_lengths):
        embedded = self.embedding(input_seq)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, input_lengths):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)

        # Initialize decoder hidden state with encoder final hidden state
        decoder_hidden = encoder_hidden

        # Use teacher forcing - feeding the target as the next input
        decoder_input = target_seq[:, 0].unsqueeze(1)  # <SOS> token

        outputs = torch.zeros(target_seq.size(0), target_seq.size(1), self.decoder.fc.out_features).to(self.device)

        for t in range(1, target_seq.size(1)):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output.squeeze(1)
            decoder_input = target_seq[:, t].unsqueeze(1)  # Teacher forcing

        return outputs


# 定义模型参数
# 将训练集和测试集合并
all_sequences = train_sequences + val_sequences + train_diagnoses + val_diagnoses

# 将整数序列展平为单个列表
all_words = [word for sequence in all_sequences for word in sequence]

# 使用集合来获取唯一的单词
vocab_set = set(all_words)

# 获取词表大小
vocab_size = len(vocab_set)

print(f"Vocabulary Size: {vocab_size}")
embedding_dim = 128
# hidden_size = 512
hidden_size = 256
output_size = vocab_size  # 输出大小等于词汇表大小
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
encoder = Encoder(vocab_size, embedding_dim, hidden_size, num_layers).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_size, output_size, num_layers).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
train_losses = []
val_losses = []
bleu_scores_list = []
rouge_scores_list = []
cider_scores_list = []
for epoch in range(num_epochs):
    # 训练模式
    model.train()

    # 训练集上的损失
    total_train_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch in train_loader:
            input_seq, target_seq, input_lengths = batch

            # 将输入和目标移至设备
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(input_seq, target_seq, input_lengths)

            # 计算损失
            # print(vocab_size,outputs.shape,outputs[:, 1:].reshape(-1, vocab_size).shape,target_seq[:, 1:].reshape(-1).shape)
            # exit(0)
            loss = criterion(outputs[:, 1:].reshape(-1, vocab_size), target_seq[:, 1:].reshape(-1))

            # 反向传播
            loss.backward()

            # 优化器更新
            optimizer.step()

            # 累计损失
            total_train_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({'Train Loss': f'{loss.item():.4f}'})

    # 平均训练集损失
    avg_train_loss = total_train_loss / len(train_loader)

    # 打印训练集损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

    # 测试模式
    model.eval()

    # 测试集上的损失和评估指标
    total_test_loss = 0.0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in val_loader:
            input_seq, target_seq, input_lengths = batch

            # 将输入和目标移至设备
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # 前向传播
            outputs = model(input_seq, target_seq, input_lengths)

            # 计算损失
            loss = criterion(outputs[:, 1:].reshape(-1, vocab_size), target_seq[:, 1:].reshape(-1))

            # 累计损失
            total_test_loss += loss.item()

            # 生成预测结果
            _, predicted_indices = outputs.max(2)
            predicted_indices = predicted_indices[:, 1:]
            # print(predicted_indices.cpu().numpy().shape)
            predictions.extend(predicted_indices.cpu().numpy())
            references.extend(target_seq[:, 1:].cpu().numpy())

    # 平均测试集损失
    avg_test_loss = total_test_loss / len(val_loader)


    references = [np.trim_zeros(ref, 'b') for ref in references]
    predictions = [np.trim_zeros(pre, 'b') for pre in predictions]
    print(predictions[0])
    print(references[0])
    # BLEU-4、ROUGE、CIDEr等评估指标
    smoothing_function = SmoothingFunction().method1  # BLEU Smoothing Function

    bleu_4_score = corpus_bleu([[ref] for ref in references], [pre.tolist() for pre in predictions],
                               smoothing_function=smoothing_function)

    # ROUGE
    references = [[' '.join(map(str, ref))] for ref in references]
    predictions = [[' '.join(map(str, pre))] for pre in predictions]
    # print(predictions[0])
    # print(references)
    # rouge_scores = rouge_score.rouge_n(predictions, references, n=4)
    rouge = Rouge()
    rouge_scores = 0
    for ref, pre in zip(references, predictions):
        # print(ref)
        # print(pre)
        # print("=============================")
        # print(ref, pre)
        rouge_scores += rouge.get_scores(ref[0], pre[0])[0]['rouge-1']['f']
    rouge_scores /= len(references)

    # CIDEr
    # cide_eval = ciderscorer.CIDEr()
    # cide_scores, _ = cide_eval.compute_score(references, predictions)
    cider_scorer = Cider()
    # 将ref和pre转换成字典
    # Compute CIDEr scores
    ref_dict = {index: value for index, value in enumerate(references)}
    pre_dict = {index: value for index, value in enumerate(predictions)}
    # print(ref_dict.keys(),pre_dict.keys())
    cider_scores, _ = cider_scorer.compute_score(ref_dict, pre_dict)

    # 打印测试集损失和评估指标
    print(
        f'Validation Loss: {avg_test_loss:.4f}, BLEU-4: {bleu_4_score:.4f}, ROUGE-1: {rouge_scores: .4f}, CIDEr: {cider_scores:.4f}')
    train_losses.append(avg_train_loss)
    val_losses.append(avg_test_loss)
    bleu_scores_list.append(bleu_4_score)
    rouge_scores_list.append(rouge_scores)
    cider_scores_list.append(cider_scores)

# Save the model state
torch.save(model.state_dict(), 'LSTM_model.pth')

# Save additional information if needed
additional_info = {
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'num_layers': num_layers,
    'device': str(device),
}

with open('LSTM_additional_info.pth', 'wb') as f:
    torch.save(additional_info, f)

plt.figure(figsize=(6, 6))
plt.subplot(221)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(222)
plt.plot(bleu_scores_list, label='BLEU-4')
# plt.plot(rouge_scores_list, label='ROUGE-1')
# plt.plot(cider_scores_list, label='CIDEr')
plt.title('Evaluation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.subplot(223)
plt.plot(cider_scores_list, label='CIDEr')
plt.title('Evaluation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.subplot(224)
plt.plot(rouge_scores_list, label='ROUGE-1')
plt.title('Evaluation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()


plt.tight_layout()
plt.show()
