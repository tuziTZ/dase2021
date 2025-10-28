import random
import sys
import time
import numpy as np


def jcd(vec1, vec2):
    intersection = 0
    union = 0
    for i in range(0, len(vec1)):
        if (vec1[i] == 1) & (vec2[i] == 1):
            intersection += 1
            union += 1
        elif (vec1[i] == 1) | (vec2[i] == 1):
            union += 1
    return intersection / (union * 1.0)


def _H(l):
    return bytes(np.array(l).byteswap().data)


class LSH:
    def __init__(self, h, r, k):
        self.h = h
        self.r = r
        self.k = k
        self.p = int(h / r)
        self.row = 0
        self.node_mat = []
        self.min_hash_mat = []
        self.prefix_forest = []
        for i in range(self.p):
            self.prefix_forest.append({})
        self.ans = []
        self.true_ans = []

    # 读取数据，给每个节点对应特征向量
    def preprocess(self):
        edge_list = []
        with open('ca-AstroPh.txt', 'r', encoding='utf-8') as fp:
            s_list = fp.read().split('\n')[:-1]
            self.row = int(s_list[0].split(' ')[1])

        for s in s_list[1:]:
            edge = s.split(' ')
            edge_list.append(edge)
        # 创建空的矩阵用来存放特征向量
        for i in range(self.row):
            self.node_mat.append([0] * self.row)

        for edge in edge_list:
            self.node_mat[int(edge[1]) - 1][int(edge[0]) - 1] = 1
            self.node_mat[int(edge[0]) - 1][int(edge[1]) - 1] = 1

    # 通过特征矩阵构建最小哈希签名矩阵
    def min_hash(self):

        for hash_num in range(0, self.h):
            ordered_index = [x for x in range(0, self.row)]
            random.shuffle(ordered_index)
            cnt = 0
            min_hash_vector = []
            for vector in self.node_mat:  # 访问每个特征向量
                for i in range(0, self.row):  # 按照哈希函数所指的顺序访问向量的每个分量，得到第一个1所循环的次数是这个向量的哈希签名(从0开始)
                    if vector[ordered_index[i]] == 1:
                        cnt = i
                        break
                min_hash_vector.append(cnt)

            self.min_hash_mat.append(min_hash_vector)

    # 将min_hash_mat分为b组，每组r行，只要两个node在任何一组有相同的向量表示，则放入同一bucket中
    def put_into_bucket(self):
        min_hash_mat_copy = self.min_hash_mat
        # 将最小哈希签名矩阵分割成待处理部分和剩余部分，每次切除r行，处理每一组放到前缀树里
        for prefix_tree_num in range(0, self.p):
            r_row_mat = min_hash_mat_copy[prefix_tree_num * self.r:prefix_tree_num * self.r + self.r]
            # 处理每一列（每个节点），映射到h/r个前缀树
            for node in range(0, self.row):
                new_list = []
                # 获取签名
                for i in range(0, self.r):
                    new_list.append(r_row_mat[i][node])
                # 签名变作tag
                tag = _H(new_list)

                if tag not in self.prefix_forest[prefix_tree_num]:
                    self.prefix_forest[prefix_tree_num][tag] = [node]
                elif node not in self.prefix_forest[prefix_tree_num][tag]:
                    self.prefix_forest[prefix_tree_num][tag].append(node)
            self.prefix_forest[prefix_tree_num] = dict(
                sorted(self.prefix_forest[prefix_tree_num].items(), key=lambda x: x[0]))
        # 以上，构建出排好序的前缀树森林，从而可以根据多探针的原理，将两侧的数据也加入到查询集中

    # 按照数据构建索引
    def make_index(self):
        ts = time.time()
        self.preprocess()
        te = time.time()
        print('数据预处理时间：' + str(te - ts) + 's')

        ts = time.time()
        self.min_hash()
        te = time.time()
        print('构建哈希签名矩阵时间：' + str(te - ts) + 's')

        ts = time.time()
        self.put_into_bucket()
        te = time.time()
        print('映射入桶时间：' + str(te - ts) + 's')

    def _query(self, node):
        # 先提取出要查询节点的最小签名
        new_list = []
        for i in range(self.h):
            new_list.append(self.min_hash_mat[i][node])
        hps = [_H(new_list[start * self.r:start * self.r + self.r]) for start in range(self.p)]
        # prefix_size=len(hps[0])
        node_list = []
        for hp, pf in zip(hps, self.prefix_forest):
            num = 0
            for j, key in enumerate(pf.keys()):
                if key == hp:
                    num = j
                    break
            l = pf[hp]
            if num != 0:
                v_list_before = list(pf.values())[num - 1]
                for n in v_list_before:
                    if n not in node_list:
                        node_list.append(n)
            if num != self.row:
                v_list_after = list(pf.values())[num + 1]
                for n in v_list_after:
                    if n not in node_list:
                        node_list.append(n)
            if len(l) > 1:
                for n in l:
                    if n not in node_list:
                        node_list.append(n)
        node_list.remove(node)
        return node_list

    # 按照用户要求输出top k
    def query(self, query_node):
        ans = {}
        node = query_node - 1
        try:
            for similar_node in self._query(node):
                ans[similar_node + 1] = (jcd(self.node_mat[node], self.node_mat[similar_node]))
            ans = sorted(ans.items(), key=lambda x: -x[1])
            if len(ans) > self.k:
                print(ans[:self.k])
                self.ans = ans[:self.k]
            else:
                print(ans)
                self.ans = ans
        except KeyError:
            print("查询失败，没有找到该节点的相似节点集")

    # 手动计算query_dict中节点的相似度
    # 验证索引时间、查询时间、空间使用和准确度
    # 对特征向量一一比较，输出实际最相似的k个节点
    def check_accuracy(self, query_node):
        time_start = time.time()
        ans = {}
        node = query_node - 1
        for similar_node in range(0, self.row):
            if similar_node != node:
                ans[similar_node + 1] = (jcd(self.node_mat[node], self.node_mat[similar_node]))
        ans = sorted(ans.items(), key=lambda x: -x[1])
        if len(ans) > self.k:
            print(ans[:self.k])
            self.true_ans = ans[:self.k]
        else:
            print(ans)
            self.true_ans = ans
        time_end = time.time()
        print('查询时间：' + str(time_end - time_start) + 's')
        precision=1 - self.err(self.true_ans, self.ans)
        recall=self.recall(self.true_ans, self.ans)
        print('本次查询的准确率为：' + str(precision))
        print('本次查询的召回率为：' + str(recall))
        return precision,recall

    def sim(self, A):
        result = 0
        for node, s in A:
            result += s
        return result / len(A)

    def err(self, A, I):
        return (self.sim(A) - self.sim(I)) / self.sim(I)

    def recall(self, A, I):
        cnt = 0
        for i in range(len(A)):
            for j in range(len(I)):
                if A[i][0] == I[j][0]:
                    cnt += 1
                    break
        return cnt / len(I)


if __name__ == '__main__':
    qn = random.sample(range(0,18771),100)  # 查询的节点序号
    qh = 100  # 最小签名矩阵的行数
    qr = [2,3,4,5,10,20]  # 每块几行
    qk = 10  # top-K

    for r in qr:
        LSH1 = LSH(qh, r, qk)  # 可调参数：哈希重排函数个数、每个分组行数、最终输出的top-k数

        time_start = time.time()
        LSH1.make_index()
        time_end = time.time()
        index_time=time_end - time_start
        print('r='+str(r)+'时，索引时间为'+str(index_time)+'s')
        query_time=0
        total_pre=0
        total_rec=0
        for n in qn:
            ts = time.time()
            LSH1.query(qn)
            te = time.time()
            query_time+=te-ts
            pre,rec=LSH1.check_accuracy(qn)
            total_pre += pre
            total_rec += rec
        print('平均查询时间为' + str(query_time/100) + 's')
        print('平均准确率为' + str(total_pre / 100) + 's')
        print('平均召回率为' + str(total_rec / 100) + 's')



    # print('LSH对象的空间占用：' + str(sys.getsizeof(LSH)))


