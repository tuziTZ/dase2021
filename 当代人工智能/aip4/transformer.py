import math

import torch.nn.functional as F
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
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

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
# 定义模型参数
# d_model = 512  # Embedding Size
d_model = 256  # Embedding Size
# d_ff = 2048  # FeedForward dimension
d_ff = 512  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

# 将训练集和测试集合并
all_sequences = train_sequences + val_sequences + train_diagnoses + val_diagnoses

# 将整数序列展平为单个列表
all_words = [word for sequence in all_sequences for word in sequence]

# 使用集合来获取唯一的单词
vocab_set = set(all_words)

# 获取词表大小
vocab_size = len(vocab_set)

# print(f"Vocabulary Size: {vocab_size}")
embedding_dim = 256
hidden_size = 512
# hidden_size = 256
output_size = vocab_size  # 输出大小等于词汇表大小
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_vocab_size = vocab_size

tgt_vocab_size = vocab_size

src_len = 150  # enc_input max sequence length
tgt_len = 80  # dec_input(=dec_output) max sequence length


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).cuda()  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len,
        # tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


# Example usage:
# vocab_size, embedding_dim, hidden_size, output_size are hyperparameters


# 创建模型
model = Transformer().cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# 训练模型
num_epochs = 20
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
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(input_seq, target_seq)

            # 计算损失
            loss = criterion(outputs, target_seq.view(-1))

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
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        with torch.no_grad():
            for batch in val_loader:
                input_seq, target_seq, input_lengths = batch

                # 将输入和目标移至设备
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                # 前向传播
                # outputs1, _, _, _ = model(input_seq[0], target_seq[0])
                # print(input_seq[0], target_seq[0],outputs1)
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(input_seq, target_seq)

                # 计算损失
                loss = criterion(outputs, target_seq.view(-1))

                # 累计损失
                total_test_loss += loss.item()

                # print(outputs.shape)
                # 生成预测结果
                # _, predicted_indices = outputs.max(2)
                predict = outputs.data.max(1, keepdim=True)[1]
                predicted_lists = []
                target_length = target_seq.shape[1]
                tmp_list = []
                count = 0
                for num in predict:
                    tmp_list.append(int(num[0]))
                    count += 1
                    if count == target_length:
                        count = 0
                        tmp = np.array(tmp_list)
                        # print(tmp)
                        predicted_lists.append(tmp)
                        tmp_list = []

                pbar.update(1)
                pbar.set_postfix({'Train Loss': f'{loss.item():.4f}'})

            # input_length=input_lengths[0]
            # print(predict[0],input_length,input_lengths[0])
            # predicted_lists=[predict[i:i + input_length].cpu().numpy() for i in range(0, len(predict), input_length)]
            # predicted_lists = [predict[start:end].cpu().numpy() for start, end in zip([0] + input_length[:-1], input_length)]
            # predicted_indices = predicted_indices[:, 1:]
            # print(predicted_indices.cpu().numpy().shape)
            # predictions.extend(predicted_indices.cpu().numpy())
            predictions.extend(predicted_lists)
            references.extend(target_seq[:, 1:].cpu().numpy())

    # 平均测试集损失
    avg_test_loss = total_test_loss / len(val_loader)

    references = [np.trim_zeros(ref, 'b') for ref in references]
    predictions = [np.trim_zeros(pre, 'b') for pre in predictions]
    # print(predictions[0])
    # print(references[0])
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
torch.save(model.state_dict(), 'transformer_model.pth')

# Save additional information if needed
additional_info = {
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'num_layers': num_layers,
    'device': str(device),
}

# with open('transformer_additional_info.pth', 'wb') as f:
#     torch.save(additional_info, f)

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
