import hashlib
import struct
import time

import numpy as np

_mersenne_prime = np.uint64((1 << 61) - 1)
_max_hash = np.uint64((1 << 32) - 1)
_hash_range = (1 << 32)


def sha1_hash32(data):
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


class MinHash:
    def __init__(self, num_perm=128, seed=1, hashfunc=sha1_hash32):
        self.seed = seed
        self.num_perm = num_perm
        self.hashfunc = hashfunc
        gen = np.random.RandomState(self.seed)
        self.permutations = np.array([
            (gen.randint(1, _mersenne_prime, dtype=np.uint64), gen.randint(0, _mersenne_prime, dtype=np.uint64)) for _
            in range(num_perm)
        ], dtype=np.uint64).T
        self.hashvalues = np.ones(num_perm, dtype=np.uint64) * _max_hash


    def update(self, b):
        hv = self.hashfunc(b)
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, _max_hash)
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def jaccard(self, other):
        return float(np.count_nonzero(self.hashvalues == other.hashvalues)) / float(len(self.hashvalues))


class MinHashLSHForest:
    def __init__(self, num_perm=128, l=8):
        # Number of prefix trees
        self.l = l
        # Maximum depth of the prefix tree
        self.k = int(num_perm / l)
        self.hashtables = []
        for _ in range(self.l):
            self.hashtables.append({})

        self.hashranges = [(i * self.k, (i + 1) * self.k) for i in range(self.l)]
        self.keys = dict()
        # This is the sorted array implementation for the prefix trees
        self.sorted_hashtables = [[] for _ in range(self.l)]

    def add(self, key, minhash):  # 传入一个MinHash变量
        self.keys[key] = [_H(minhash.hashvalues[start:end])
                          for start, end in self.hashranges]  # 长度为8，把最小签名向量切成了8段，每一段投射了一个巨长的值
        for H, hashtable in zip(self.keys[key], self.hashtables):
            if H not in hashtable:
                hashtable[H] = [key]
            else:
                hashtable[H].append(key)  # 有时，2和3会装到一个里，有时会分开装（到H生成的桶里）

    def index(self):
        for i, hashtable in enumerate(self.hashtables):
            self.sorted_hashtables[i] = [H for H in hashtable.keys()]
            self.sorted_hashtables[i].sort()

    def _query(self, minhash, r):
        hps = [_H(minhash.hashvalues[start:start + r])
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

    def query(self, node, minhash, k):
        results = set()
        r = self.k
        while r > 0:
            for key in self._query(minhash, r):
                results.add(key)
                if len(results) >= k + 1:
                    result = list(results)
                    result.remove(node)
                    return result
            r -= 1
        result = list(results)
        result.remove(node)
        return result


def preprocessing():
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
    return node_mat


if __name__ == '__main__':
    ts=time.time()
    node_mat = preprocessing()
    te=time.time()
    print('数据预处理时间',te-ts)

    ts = time.time()
    forest = MinHashLSHForest()
    min_hash_list = {}
    for node, value in node_mat.items():
        m = MinHash()
        for element in value:
            m.update(element.encode('utf-8'))
        forest.add(node, m)
        min_hash_list[node] = m
    forest.index()
    te = time.time()
    print('索引构建时间', te - ts)

    ts = time.time()
    q = '8'
    m = min_hash_list[q]
    result = forest.query(q, m, 10)

    jcd_list = []
    for node in result:
        jcd_list.append(m.jaccard(min_hash_list[node]))
    l = list(zip(result, jcd_list))
    l.sort(key=lambda x:x[1],reverse=True)
    print(l)
    te = time.time()
    print('查询时间', te - ts)
