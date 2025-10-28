import time

import numpy as np


def NMF(shape, mu, Data, K=10, times=100, alpha=0.005, beta=0.05):
    W = np.random.rand(shape[0], K)
    H = np.random.rand(shape[1], K).T
    b = 5 * np.random.rand(shape[0])
    d = 5 * np.random.rand(shape[1])
    rmse_list = []
    mae_list = []
    for t in range(times):
        print('-----------------------------')
        print('开始第', t, '次迭代')
        ts = time.time()
        for data in Data:
            i = data[0]-1
            j = data[1]-1
            eij = data[2] - np.dot(W[i], H[:, j]) - mu - b[i] - d[j]
            for k in range(K):
                W[i][k] = W[i][k] + alpha * (eij * H[k][j] - beta * W[i][k])
                H[k][j] = H[k][j] + alpha * (eij * W[i][k] - beta * H[k][j])
                b[i] = b[i] + alpha * (eij - beta * b[i])
                d[j] = d[j] + alpha * (eij - beta * d[j])
        te = time.time()
        print(te - ts)
        V_new = np.dot(W, H)
        rmse = 0
        mae = 0
        cnt = 0
        for data in Data:
            i = data[0]-1
            j = data[1]-1
            rmse += pow((data[2] - V_new[i][j]), 2)
            mae += abs(data[2] - V_new[i][j])
            cnt += 1
        rmse = np.sqrt(rmse / cnt)
        mae = mae / cnt
        print(' rmse=' + str(rmse) + ' mae=' + str(mae))
        rmse_list.append(rmse)
        mae_list.append(mae)
    print(rmse_list)
    print(mae_list)
    return V_new


def parse_rating(line):
    rating_data = line.strip().split("::")
    return [int(rating_data[0]), int(rating_data[1]), float(rating_data[2])]


if __name__ == '__main__':
    query_num = 6
    x = []
    with open("./ml-1m/ratings.dat", 'r') as file:
        data = [parse_rating(line) for line in file]
    mean = np.mean(data, axis=0)[2]
    max_list = list(np.max(data, axis=0))
    n = int(max_list[0])
    m = int(max_list[1])
    shape = (n, m)
    V_new = NMF(shape, mean, data)
    V = np.zeros((int(n), int(m)))
    for row in x:
        V[row[0] - 1][row[1] - 1] = row[2]
    old_list = list(V[query_num])
    new_list = list(V_new[query_num])
    max = 0
    index = 0
    for i in range(len(old_list)):
        if old_list[i] == 0 and new_list[i] > max:
            max = new_list[i]
            index = i
    print("为用户" + str(query_num) + '推荐电影' + str(index) + ',预测评分为' + str(max))
