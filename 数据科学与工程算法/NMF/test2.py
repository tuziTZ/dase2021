import numpy as np

def myNMF(V, k, epsilon, itermax):
    # pre-work
    n, m = V.shape
    # initialize W and H
    W_init = np.random.rand(n, k)
    H_init = np.random.rand(k, m)
    H_old = H_init
    W_old = W_init
    H_new = np.zeros((k, m))
    W_new = np.zeros((n, k))
    # some prepare variables
    dist = np.zeros(itermax)
    count = 2
    dist[count] = 100
    error = np.finfo(float).max

    # iterate
    while error >= epsilon:
        # update matrix H
        Hcoematx_up = np.dot(W_old.T, V)
        Hcoematx_dn = np.dot(np.dot(W_old.T, W_old), H_old)
        for i in range(k):
            for j in range(m):
                if Hcoematx_dn[i, j] == 0:
                    H_new[i, j] = H_old[i, j]
                else:
                    H_new[i, j] = H_old[i, j] * Hcoematx_up[i, j] / Hcoematx_dn[i, j]
        # update matrix W
        Wcoematx_up = np.dot(V, H_old.T)
        Wcoematx_dn = np.dot(np.dot(W_old, H_old), H_old.T)
        for i in range(n):
            for j in range(k):
                if Wcoematx_dn[i, j] == 0:
                    W_new[i, j] = W_old[i, j]
                else:
                    W_new[i, j] = W_old[i, j] * Wcoematx_up[i, j] / Wcoematx_dn[i, j]
        # calculate the difference between two iteration approximation matrices
        dist[count] = np.sum(np.sum((W_new.dot(H_new) - W_old.dot(H_old))**2))
        error = np.abs(dist[count] - dist[count-1])
        if count % 1000 == 0:
            print(f'{count}轮迭代误差为{dist[count]}.')
        # The results of this round of iteration
        if count - 1 == itermax:
            print(f'{itermax}轮迭代已毕，误差仍未收敛.')
            break
        # prepare for the next round of round
        H_old = H_new
        W_old = W_new
        count += 1

    return W_new, H_new, count, dist[count]