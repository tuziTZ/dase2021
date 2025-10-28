import random
import time


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


class LSH:
    def __init__(self,h,r):
        self.h = h
        self.r = r
        self.k = 0
        self.row = 0
        self.node_mat = []
        self.min_hash_mat = []
        self.bucket_list = []
        self.query_dict = {}
        self.true_ans = []
        self.ans = []

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
        # 将最小哈希签名矩阵分割成待处理部分和剩余部分，每次切除r行
        while len(min_hash_mat_copy) > self.r:
            hashBuckets = {}
            r_row_mat = min_hash_mat_copy[:self.r]
            min_hash_mat_copy = min_hash_mat_copy[self.r + 1:]
            # 处理每一列（每个节点），用哈希函数MD5得到映射进的桶
            for node in range(0, self.row):
                new_list = []
                # 将每一行的对应签名插入
                for i in range(0, self.r):
                    new_list.append(r_row_mat[i][node])
                # hashObj = hashlib.md5()
                # # band = str(new_list)
                # band = " ".join(map(str, new_list))
                # hashObj.update(band.encode())
                # tag = hashObj.hexdigest()
                tag = str(new_list)
                if tag not in hashBuckets:
                    hashBuckets[tag] = [node]
                elif node not in hashBuckets[tag]:
                    hashBuckets[tag].append(node)
            # 每一组操作完毕后，忽略列表长度为1的，记录共同出现的节点
            for key, value in hashBuckets.items():
                if len(value) >= 2:
                    # 将节点之间的相关度关系存储在query_dict中
                    for i in value:  # key值
                        for j in value:  # 将要加入列表的value值
                            if i not in self.query_dict:
                                self.query_dict[i] = [j]
                            elif j not in self.query_dict[i]:
                                self.query_dict[i].append(j)

    # 按照数据构建索引
    # def make_index(self):
    #     ts = time.time()
    #     self.preprocess()
    #     te = time.time()
    #     print('数据预处理时间：' + str(te - ts) + 's')
    #
    #     ts = time.time()
    #     self.min_hash()
    #     te = time.time()
    #     print('构建哈希签名矩阵时间：' + str(te - ts) + 's')
    #
    #     ts = time.time()
    #     self.put_into_bucket()
    #     te = time.time()
    #     print('映射入桶时间：' + str(te - ts) + 's')

    # 按照用户要求输出top k
    def query(self, query_node, k):
        self.k = k
        ans = {}
        node = query_node - 1
        try:
            for similar_node in self.query_dict[node]:
                if similar_node != node:
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
        return ans

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
        return self.true_ans

def pre( A, I):
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
    return 1 - (sim_I - sim_A) / sim_I

def recall( A, I):
    cnt = 0
    for i in range(len(A)):
        for j in range(len(I)):
            if A[i][0] == I[j][0]:
                cnt += 1
                break
    return cnt / len(I)


if __name__ == '__main__':
    LSH1 = LSH(30, 2)

    ts = time.time()
    LSH1.preprocess()
    te = time.time()
    print('数据预处理时间', te - ts)

    ts = time.time()
    LSH1.min_hash()
    te = time.time()
    print('最小哈希签名构建时间', te - ts)

    ts = time.time()
    LSH1.put_into_bucket()
    te = time.time()
    print('森林构建时间', te - ts)

    ts = time.time()
    result = LSH1.query(8, 10)
    print(result)
    te = time.time()
    print('查询时间', te - ts)

    ts = time.time()
    acc_result = LSH1.check_accuracy(8, 10)
    print(acc_result)
    te = time.time()
    print('查询时间', te - ts)

    print(recall(result, acc_result))
    print(pre(result, acc_result))

    # qn = random.sample(range(1, 18772), 10)  # 查询的节点序号
    # qh = [30, 60, 90]  # 最小签名矩阵的行数
    # qr = [2, 3, 4, 5, 10]  # 每块几行
    # qk = 10  # top-K

        # LSH1 = LSH()  # 可调参数：哈希重排函数个数、每个分组行数、最终输出的top-k数
        # ts = time.time()
        # LSH1.preprocess()
        # te = time.time()
        # fp.write('数据预处理时间为 ' + str(te - ts) + 's\n')
        # for h in qh:
        #     fp.write('重排函数为 ' + str(h) + '\n')
        #     ts = time.time()
        #     LSH1.min_hash(h)
        #     te = time.time()
        #     fp.write('最小哈希签名矩阵构建时间为 ' + str(te - ts) + 's\n')
        #     for r in qr:
        #         fp.write('每块行数为 ' + str(r) + '\n')
        #         ts = time.time()
        #         LSH1.put_into_bucket(r)
        #         te = time.time()
        #         fp.write('映射入桶时间为 ' + str(te - ts) + 's\n')
        #         query_time = 0
        #         total_pre = 0
        #         total_rec = 0
        #         for n in qn:
        #             # fp.write('查询节点为 ' + str(n) + '\n')
        #             ts = time.time()
        #             LSH1.query(n, qk)
        #             te = time.time()
        #             query_time += te - ts
        #             # fp.write('查询时间为 ' + str(te - ts) + 's\n')
        #             pre, rec = LSH1.check_accuracy(n, qk)
        #             total_pre += pre
        #             total_rec += rec
        #         fp.write('平均查询时间为 ' + str(query_time / 10) + '\n')
        #         fp.write('平均准确率为 ' + str(total_pre / 10) + '\n')
        #         fp.write('平均召回率为 ' + str(total_rec / 10) + '\n')
