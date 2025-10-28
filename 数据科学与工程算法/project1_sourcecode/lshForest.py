import hashlib
import struct
import numpy as np
import time

_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = (1 << 32)


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


def hashfunc(data):
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


def _H(hs):
    return bytes(hs.byteswap().data)


def _binary_search(n, func):
    i, j = 0, n
    while i < j:
        h = int(i + (j - i) / 2)
        if not func(h):
            i = h + 1
        else:
            j = h
    return i


class LSH:
    def __init__(self,h,l):
        self.h = h
        self.node_mat = {}
        self.min_hash_list = {}
        self.l = l
        gen = np.random.RandomState(1)
        self.permutations = np.array([
            (gen.randint(1, _mersenne_prime, dtype=np.uint64), gen.randint(0, _mersenne_prime, dtype=np.uint64)) for _
            in range(self.h)
        ], dtype=np.uint64).T
        self.k = 0
        self.hashtables = []
        for _ in range(self.l):
            self.hashtables.append({})
        self.hashranges = []
        self.keys = dict()
        self.sorted_hashtables = [[] for _ in range(self.l)]

    def jaccard(self, m1, m2):
        v1 = self.min_hash_list[m1]
        v2 = self.min_hash_list[m2]
        return float(np.count_nonzero(v1 == v2)) / float(len(v1))

    def preprocessing(self):
        edge_list = []
        with open('ca-AstroPh.txt', 'r', encoding='utf-8') as fp:
            s_list = fp.read().split('\n')[1:-1]

        for s in s_list:
            edge = s.split(' ')
            edge_list.append(edge)
        # 创建空的字典用来存放特征向量
        node_mat = {}
        for edge in edge_list:
            if edge[1] not in node_mat:
                node_mat[edge[1]] = [edge[0]]
            else:
                node_mat[edge[1]].append(edge[0])
            if edge[0] not in node_mat:
                node_mat[edge[0]] = [edge[1]]
            else:
                node_mat[edge[0]].append(edge[1])
        self.node_mat = node_mat

    def min_hash(self):
        for key, value in self.node_mat.items():
            hashvalues = np.ones(self.h, dtype=np.uint64) * _max_hash
            for b in value:
                hv = hashfunc(b.encode('utf-8'))
                a, b = self.permutations
                phv = np.bitwise_and((a * hv + b) % _mersenne_prime, _max_hash)
                hashvalues = np.minimum(phv, hashvalues)

            self.min_hash_list[key] = hashvalues

    def forest(self):
        self.k = int(self.h / self.l)
        self.hashranges = [(i * self.k, (i + 1) * self.k) for i in range(self.l)]
        for key, hashvalues in self.min_hash_list.items():
            self.keys[key] = [_H(hashvalues[start:end])
                              for start, end in self.hashranges]
            for H, hashtable in zip(self.keys[key], self.hashtables):
                if H not in hashtable:
                    hashtable[H] = [key]
                else:
                    hashtable[H].append(key)
        for i, hashtable in enumerate(self.hashtables):
            self.sorted_hashtables[i] = [H for H in hashtable.keys()]
            self.sorted_hashtables[i].sort()

    def _query(self, node, r):
        hashvalues = self.min_hash_list[node]
        hps = [_H(hashvalues[start:start + r])
               for start, _ in self.hashranges]
        prefix_size = len(hps[0])
        for ht, hp, hashtable in zip(self.sorted_hashtables, hps, self.hashtables):
            i = _binary_search(len(ht), lambda x: ht[x][:prefix_size] >= hp)
            if i < len(ht) and ht[i][:prefix_size] == hp:
                j = i
                while j < len(ht) and ht[j][:prefix_size] == hp:
                    for key in hashtable[ht[j]]:
                        yield key
                    j += 1

    def query(self, node, k):
        results = set()
        r = self.k
        while r > 0:  # 查询r棵树
            for key in self._query(node, r):
                results.add(key)
                if len(results) >= k + 1:
                    result = list(results)
                    result.remove(node)
                    jcd_list = []
                    for n in result:
                        jcd_list.append(self.jaccard(node, n))
                    _result = list(zip(result, jcd_list))
                    _result.sort(key=lambda x: x[1], reverse=True)
                    return _result
            r -= 1
        result = list(results)
        result.remove(node)
        jcd_list = []
        for n in result:
            jcd_list.append(self.jaccard(node, n))
        _result = list(zip(result, jcd_list))
        _result.sort(key=lambda x: x[1], reverse=True)
        return _result

    def check_accuracy(self, query_node, k):
        acc_list = []
        for key in self.min_hash_list.keys():
            value = self.jaccard(query_node, key)
            acc_list.append((key, value))
        acc_list.sort(key=lambda x: x[1], reverse=True)
        del (acc_list[0])
        return acc_list[:k]


if __name__ == '__main__':
    LSH1 = LSH(128,8)


    ts = time.time()
    LSH1.preprocessing()
    te = time.time()
    print('数据预处理时间', te - ts)

    ts = time.time()
    LSH1.min_hash()
    te = time.time()
    print('最小哈希签名构建时间', te - ts)

    ts = time.time()
    LSH1.forest()
    te = time.time()
    print('森林构建时间', te - ts)

    ts = time.time()
    result = LSH1.query('8', 10)
    print(result)
    te = time.time()
    print('查询时间', te - ts)

    ts = time.time()
    acc_result = LSH1.check_accuracy('8', 10)
    print(acc_result)
    te = time.time()
    print('查询时间', te - ts)

    print(recall(result,acc_result))
    print(pre(result,acc_result))
