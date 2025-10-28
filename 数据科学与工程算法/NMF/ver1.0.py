import numpy as np


def NMF(V,K=10, times=100, alpha=0.005, beta=0.05):
    shape=V.shape
    W = np.random.rand(shape[0], K)
    H = np.random.rand(shape[1], K).T
    for t in range(times):
        print('-----------------------------')
        print('开始第', t, '次迭代')
        for i in range(shape[0]):
            for j in range(shape[1]):
                if V[i][j] > 0:
                    eij = V[i][j] - np.dot(W[i], H[:, j])
                    for k in range(K):
                        W[i][k] = W[i][k] + alpha * (eij * H[k][j] - beta * W[i][k])
                        H[k][j] = H[k][j] + alpha * (eij * W[i][k] - beta * H[k][j])
        V_new = np.dot(W, H)
        rmse = 0
        mae = 0
        cnt = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if V[i][j] > 0:
                    rmse += pow((V[i][j] - V_new[i][j]), 2)
                    mae += abs(V[i][j] - V_new[i][j])
                    cnt += 1
        rmse = np.sqrt(rmse / cnt)
        mae = mae / cnt
        print(' rmse=' + str(rmse)+' mae='+str(mae))
    return V_new


if __name__ == '__main__':
    query_num = 6
    x = []
    with open("./ml-1m/ratings.dat", 'r') as file:
        reader = file.readlines()
    for r in reader:
        r_list = r.split("::")
        x.append([int(r_list[0]), int(r_list[1]), float(r_list[2])])
    data = np.array(x, dtype=float)
    max_list = list(np.max(data, axis=0))
    n = max_list[0]
    m = max_list[1]
    V = np.zeros((int(n), int(m)))
    for row in x:
        V[int(row[0]) - 1][int(row[1]) - 1] = float(row[2])

    V_new = NMF(V)
    old_list = list(V[query_num])
    new_list = list(V_new[query_num])
    max = 0
    index = 0
    for i in range(len(old_list)):
        if old_list[i] == 0 and new_list[i] > max:
            max = new_list[i]
            index = i
    print("为用户" + str(query_num) + '推荐电影' + str(index) + ',预测评分为' + str(max))
