import sys
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

def load_model():
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model

def get_embedding(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()

def main():
    tokenizer, model = load_model()
    query_word = sys.argv[1]  # 获取命令行传入的词
    # 这里只是示例，具体实现需要添加找到最相似的词的逻辑
    print(query_word)  # 打印查询词来确认接收到了

if __name__ == "__main__":
    main()
