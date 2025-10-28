import hashlib
import random
import sys
import threading
import time
import numpy as np
import kmeans

min_hash_vector = []
lock = threading.RLock()


def _min_hash(LSH, hash_list):
    for hash_num in hash_list:  # 2-这些哈希函数的操作可以是并行的，把他们放在不同线程里
        global min_hash_vector
        lock.acquire()
        ordered_index = [x for x in range(0, LSH.row)]
        cnt = 0
        min_hash_vector = []
        # 重排规则是洗牌算法生成的序列
        for num_max in reversed(range(0, LSH.row)):
            num_chose = random.randint(0, num_max)
            tmp = ordered_index[num_chose]
            ordered_index[num_chose] = ordered_index[num_max]
            ordered_index[num_max] = tmp
        ordered_dict = dict(zip([x for x in range(0, LSH.row)], ordered_index))

        for vector in LSH.node_mat:  # 访问每个特征向量
            for i in range(0, LSH.row):  # 按照哈希函数所指的顺序访问向量的每个分量，得到第一个1所循环的次数是这个向量的哈希签名(从0开始)
                if vector[ordered_dict[i]] == 1:  # 1-这里可以去除重复的重排方式，等我研究研究
                    cnt = i
                    break
            min_hash_vector.append(cnt)
        LSH.min_hash_mat.append(min_hash_vector)
        lock.release()


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
    # return ' '.join(map(str,l))
    return bytes(np.array(l).byteswap().data)


class LSH:
    def __init__(self, h, r, k, clusters, error):
        self.h = h
        self.r = r
        self.k = k
        self.p = int(h / r)
        self.clusters = clusters
        self.error = error
        self.row = 0
        self.node_mat = []
        self.node_mat_cluster = []
        self.min_hash_mat = []
        self.prefix_forest = []
        for i in range(self.p):
            self.prefix_forest.append({})
        self.query_dict = {}

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

    # 将特征向量分成多个集群
    def separate(self):
        # dataSet=dict(zip(range(0,self.row),self.node_mat))

        centroids, cluster = kmeans.kmeans(self.node_mat, self.clusters, self.error)
        for i in cluster:
            print(len(i))

    # 通过特征矩阵构建最小哈希签名矩阵
    def min_hash(self, thread_num):
        hash_list_all = range(0, self.h)
        thread_list = []
        step = int(self.h / thread_num)
        for t in range(thread_num):
            thread_list.append(threading.Thread(target=_min_hash, args=(self, hash_list_all[t * step:t * step + step])))

        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()

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
                # tag=' '.join(map(str, new_list))
                # tag=bytes(new_list)
                tag = _H(new_list)

                if tag not in self.prefix_forest[prefix_tree_num]:
                    self.prefix_forest[prefix_tree_num][tag] = [node]
                elif node not in self.prefix_forest[prefix_tree_num][tag]:
                    self.prefix_forest[prefix_tree_num][tag].append(node)
            self.prefix_forest[prefix_tree_num] = dict(
                sorted(self.prefix_forest[prefix_tree_num].items(), key=lambda x: x[0]))
        # 以上，构建出排好序的前缀树森林，从而可以根据多探针的原理，将两侧的数据也加入到查询集中
        # print(self.prefix_forest)

    # 按照数据构建索引
    def make_index(self):
        ts = time.time()
        self.preprocess()
        te = time.time()
        print('数据预处理时间：' + str(te - ts) + 's')

        ts = time.time()
        self.min_hash(6)
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
            # print(hp,pf[0])
            # i= _binary_search(len(pf), lambda x: pf[x] >= hp)
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
        # 明天再实现一种前缀树的写法：不按照条块划分，而是顺序取，使得可以在取到10个之后立马返回，省时间
        # 失败了
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
            else:
                print(ans)
        except KeyError:
            print("查询失败，没有找到该节点的相似节点集")

    # 手动计算query_dict中节点的相似度

    # 将向量hash到桶里的函数,要保证相同向量hash值相同，把只有一个值的桶删掉(目前暂用md5)
    # def hash_bucket(self,vector):

    # 验证索引时间、查询时间、空间使用和准确度
    # 对特征向量一一比较，输出实际最相似的k个节点
    def check_accuracy(self, query_node):
        ans = {}
        node = query_node - 1
        for similar_node in range(0, self.row):
            if similar_node != node:
                ans[similar_node + 1] = (jcd(self.node_mat[node], self.node_mat[similar_node]))
        ans = sorted(ans.items(), key=lambda x: -x[1])
        if len(ans) > self.k:
            print(ans[:self.k])
        else:
            print(ans)


if __name__ == '__main__':
    qn = 8
    qh = 30
    qr = 2
    qk = 10
    LSH1 = LSH(qh, qr, qk, clusters=3, error=10)  # 可调参数：哈希重排函数个数、每个分组行数、最终输出的top-k数
    # 观察数据集中相似度较高的top-k一般是多少，调整参数使得这些值在S曲线的右侧（更高概率被加入）
    # 画出S曲线
    # draw_S(qh, qr)
    # print(LSH1.prefix_forest)
    # ts = time.time()
    # LSH1.preprocess()
    # te = time.time()
    # print('数据预处理时间：' + str(te - ts) + 's')
    #
    # ts = time.time()
    # LSH1.separate()
    # te = time.time()
    # print('聚类时间：' + str(te - ts) + 's')

    time_start = time.time()
    LSH1.make_index()
    time_end = time.time()
    print('索引时间：' + str(time_end - time_start) + 's')

    time_start = time.time()
    LSH1.query(qn)
    time_end = time.time()
    print('查询时间：' + str(time_end - time_start) + 's')
    print('LSH对象的空间占用：' + str(sys.getsizeof(LSH)))

    LSH1.check_accuracy(qn)
