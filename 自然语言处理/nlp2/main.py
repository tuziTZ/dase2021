# 分词：将连续的字序列按照规范生成词序列。
#
# 词频统计：将第一步得到的分词结果根据出现频率进行统计。
#
# 绘制词云：对高频关键词的可视化表达。
# 数据集结构：article/1.txt,2.txt,...,20.txt
import math

# 分词工具：国内比较流行的中文分词工具有jieba、SnowNLP、THULAC、NLPIR等，上述分词工具都已经在github上开源。
# 词频统计：python提供了许多第三方库来帮助完成词频统计，包括collections库、pandas库等。
# 绘制词云：wordcloud库、pyecharts库的WordCloud、stylecloud库等
import jieba
import os


# 读取文章
def read_article(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text_without_newlines = text.replace('\n', ' ')
    return text_without_newlines


# 分词
def segment_text(text, stop_words):
    words = jieba.lcut(text)

    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


# 对所有文章进行分词
def segment_articles(directory, stop_words):
    segmented_articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            text = read_article(filepath)
            segmented_text = segment_text(text, stop_words)
            segmented_articles.append(segmented_text)
    return segmented_articles


# 分词结果保存到文件
# def save_segmented_articles(segmented_articles, output_directory):
#     for idx, segmented_text in enumerate(segmented_articles):
#         with open(os.path.join(output_directory, f"{idx+1}.seg"), 'w', encoding='utf-8') as file:
#             file.write(" ".join(segmented_text))


# save_segmented_articles(segmented_articles, output_directory)
from collections import Counter


# 计算TF，表示词word在当前document中 出现频率
def calculate_tf(word, document):
    return document.count(word) / len(document)


# 计算IDF，表示词word在所有documents中的常见程度
def calculate_idf(word, documents):
    num_documents_containing_word = sum(1 for document in documents if word in document)
    return math.log(len(documents) / (1 + num_documents_containing_word))


# 计算TF-IDF
def calculate_tfidf(word, document, documents):
    return calculate_tf(word, document) * calculate_idf(word, documents)


# 统计词频并转换为TF-IDF
def count_word_frequency(segmented_articles):
    word_frequency = Counter()
    documents = len(segmented_articles)

    for segmented_text in segmented_articles:
        # 每篇文章的分词结果
        for word in set(segmented_text):
            tfidf = calculate_tfidf(word, segmented_text, segmented_articles)
            word_frequency[word] += tfidf

    return word_frequency


# 加载停用词库
def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return set(stopwords)


# 过滤词频统计结果
def filter_word_frequency(word_frequency, stopwords):
    filtered_word_frequency = {word: freq for word, freq in word_frequency.items() if word not in stopwords}
    return filtered_word_frequency


# 打印词频
def print_word_frequency(word_frequency):
    for word, freq in word_frequency.most_common():
        # print(f"{word}: {freq}")
        print(word)


# 分词处理
input_directory = "article"
stop_words = load_stopwords('cn_stopwords.txt')
segmented_articles = segment_articles(input_directory, stop_words)

word_frequency = count_word_frequency(segmented_articles)

print_word_frequency(word_frequency)
# word_frequency_dict = filter_word_frequency(word_frequency, stopwords)
# print(word_frequency_dict)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# 绘制词云
def draw_word_cloud(word_frequency):
    wordcloud = WordCloud(font_path='SimSun.ttf', stopwords=STOPWORDS, width=800, height=400, background_color='white',
                          # 背景颜色
                          colormap='viridis', ).generate_from_frequencies(word_frequency)
    # plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# 绘制词云
draw_word_cloud(word_frequency)
