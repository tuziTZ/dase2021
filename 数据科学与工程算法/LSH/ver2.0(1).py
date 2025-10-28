import random
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
    def __init__(self):
        self.h = 0
        self.r = 0
        self.p = 0
        self.k = 0
        self.row = 0
        self.node_mat = []
        self.min_hash_mat = []
        self.bucket_list = []
        self.query_dict = {}
        self.true_ans = []
        self.ans = []
        self.prefix_forest = []

    # 随机生成无向图数据

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

        # for node in range(1, self.row + 1):
        #     node_vector = [0] * self.row
        #     for edge in edge_list:
        #         if int(edge[0]) == node:
        #             node_vector[int(edge[1]) - 1] = 1
        #         elif int(edge[1]) == node:
        #             node_vector[int(edge[0]) - 1] = 1
        #     self.node_mat.append(node_vector)

    # 通过特征矩阵构建最小哈希签名矩阵
    # def min_hash(self):
    #
    #     for hash_num in range(0, self.h):  # 2-这些哈希函数的操作可以是并行的，把他们放在不同线程里，最后再把哈希签名矩阵按顺序合成一个
    #         ordered_index = [x for x in range(0, self.row)]
    #         cnt = 0
    #         min_hash_vector = []
    #         每次循环生成一个哈希重排函数
    #         random.gauss(0, 1)
    #         a = random.randint(1, 100)
    #         b = random.randint(1, self.row - 1)  # 依据来自第二篇论文
    #         for vector in self.node_mat:  # 访问每个特征向量
    #             for i in range(0, self.row):  # 按照哈希函数所指的顺序访问向量的每个分量，得到第一个1所循环的次数是这个向量的哈希签名(从0开始)
    #                 if vector[(i * a + b) % self.row] == 1:  # 1-这里可以去除重复的重排方式，等我研究研究
    #                     cnt = i
    #                     break
    #             min_hash_vector.append(cnt)
    #
    # # 重排规则是洗牌算法生成的序列
    # for num_max in reversed(range(0, self.row)):
    #     num_chose = random.randint(0, num_max)
    #     tmp = ordered_index[num_chose]
    #     ordered_index[num_chose] = ordered_index[num_max]
    #     ordered_index[num_max] = tmp
    #
    #         for vector in self.node_mat:  # 访问每个特征向量
    #             for i in range(0, self.row):  # 按照哈希函数所指的顺序访问向量的每个分量，得到第一个1所循环的次数是这个向量的哈希签名(从0开始)
    #                 if vector[ordered_index[i]] == 1:  # 1-这里可以去除重复的重排方式，等我研究研究
    #                     cnt = i
    #                     break
    #             min_hash_vector.append(cnt)
    #
    #         self.min_hash_mat.append(min_hash_vector)
    def min_hash(self, h):
        self.h = h
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
    def put_into_bucket(self, r):
        self.r=r
        self.p = int(self.h / self.r)
        for i in range(self.p):
            self.prefix_forest.append({})
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



    # 按照用户要求输出top k
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
        return node_list

    # 按照用户要求输出top k
    def query(self, query_node, k):
        self.k = k
        ans = {}
        node = query_node - 1
        try:
            for similar_node in self._query(node):
                ans[similar_node + 1] = (jcd(self.node_mat[node], self.node_mat[similar_node]))
            ans = sorted(ans.items(), key=lambda x: -x[1])
            if len(ans) > self.k:
                # print(ans[:self.k])
                self.ans = ans[:self.k]
            else:
                # print(ans)
                self.ans = ans
        except KeyError:
            print("查询失败，没有找到该节点的相似节点集")

    # 手动计算query_dict中节点的相似度

    # 将向量hash到桶里的函数,要保证相同向量hash值相同，把只有一个值的桶删掉(目前暂用md5)
    # def hash_bucket(self,vector):

    # 验证索引时间、查询时间、空间使用和准确度
    # 对特征向量一一比较，输出实际最相似的k个节点
    def check_accuracy(self, query_node, k):
        self.k = k
        ans = {}
        node = query_node - 1
        for similar_node in range(0, self.row):
            if similar_node != node:
                ans[similar_node + 1] = (jcd(self.node_mat[node], self.node_mat[similar_node]))
        ans = sorted(ans.items(), key=lambda x: -x[1])
        if len(ans) > self.k:
            # print(ans[:self.k])
            self.true_ans = ans[:self.k]
        else:
            # print(ans)
            self.true_ans = ans
        if len(self.ans) == 0:
            return 0, 0
        pre = self.pre(self.true_ans, self.ans)
        recall = self.recall(self.true_ans, self.ans)
        return pre, recall

    def pre(self, A, I):
        result = 0
        for node, s in A:
            result += s
        sim_A = result / len(A)
        result = 0
        cnt = 0
        for node, s in I:
            result += s
            cnt += 1
            if cnt == len(A):
                break
        sim_I = result / len(A)
        return 1 - (sim_A - sim_I) / sim_I

    def recall(self, A, I):
        cnt = 0
        for i in range(len(A)):
            for j in range(len(I)):
                if A[i][0] == I[j][0]:
                    cnt += 1
                    break
        return cnt / len(I)


if __name__ == '__main__':
    qn = random.sample(range(1, 18772), 10)  # 查询的节点序号
    qh = [30, 60, 90]  # 最小签名矩阵的行数
    qr = [2, 3, 4, 5, 10]  # 每块几行
    qk = 10  # top-K
    with open('ver2.0.txt', 'w', encoding='utf-8') as fp:
        LSH1 = LSH()  # 可调参数：哈希重排函数个数、每个分组行数、最终输出的top-k数
        ts = time.time()
        LSH1.preprocess()
        te = time.time()
        fp.write('数据预处理时间为 ' + str(te - ts) + 's\n')
        for h in qh:
            fp.write('重排函数为 ' + str(h) + '\n')
            ts = time.time()
            LSH1.min_hash(h)
            te = time.time()
            fp.write('最小哈希签名矩阵构建时间为 ' + str(te - ts) + 's\n')
            for r in qr:
                fp.write('每块行数为 ' + str(r) + '\n')
                ts = time.time()
                LSH1.put_into_bucket(r)
                te = time.time()
                fp.write('映射入桶时间为 ' + str(te - ts) + 's\n')
                query_time = 0
                total_pre = 0
                total_rec = 0
                for n in qn:
                    # fp.write('查询节点为 ' + str(n) + '\n')
                    ts = time.time()
                    LSH1.query(n, qk)
                    te = time.time()
                    query_time += te - ts
                    # fp.write('查询时间为 ' + str(te - ts) + 's\n')
                    pre, rec = LSH1.check_accuracy(n, qk)
                    total_pre += pre
                    total_rec += rec
                fp.write('平均查询时间为 ' + str(query_time / 10) + '\n')
                fp.write('平均准确率为 ' + str(total_pre / 10) + '\n')
                fp.write('平均召回率为 ' + str(total_rec / 10) + '\n')
