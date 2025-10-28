import numpy as np
import matplotlib.pyplot as plt


def func_S(s, r, b):
    P = 1 - pow((1 - pow(s, r)), b)
    return P


# def draw_S(h, r):
#     b = int(h / r)
#     s = np.arange(0, 1, 0.01)
#     P = []
#     for x in s:
#         y = func_S(x, r, b)
#         P.append(y)
#     plt.plot(s, P, label='S-curve')
#     plt.xlabel('h='+str(h)+', r='+str(r))
#     plt.legend()
#     plt.show()

# def draw_S(b, r):
#     s = np.arange(0, 1, 0.01)
#     P = []
#     for x in s:
#         y = func_S(x, r, b)
#         P.append(y)
#     plt.plot(s, P, label='S-curve')
#     plt.xlabel('b='+str(b)+', r='+str(r))
#     plt.legend()
#     plt.show()

def draw_S():
    r_list=[2,3,4,5,10]
    h_list=[30,60,90]
    color_list=['red','green','blue']
    for r in r_list:
        for h,c in zip(h_list,color_list):
            b=h//r
            s = np.arange(0, 1, 0.01)
            P = []
            for x in s:
                y = func_S(x, r, b)
                P.append(y)
            plt.plot(s, P, label='h='+str(h)+', r='+str(r),color=c)
            plt.legend()
    plt.show()

def draw():
    # ver1.0
    # 数据预处理时间为 2.4405102729797363s
    # 重排函数与最小哈希签名矩阵构建时间的关系
    ax1 = plt.subplot(141)
    h_list=[30,60,90]
    min_hash_time=[80.61964130401611,117.58416199684143,213.6061246395111]
    plt.plot(h_list,min_hash_time,label='x-h,y-minhash time')
    # plt.legend()
    # plt.show()

    ax1 = plt.subplot(142)
    query_time=[0.04633188247680664,0.07162914276123047,0.11865706443786621]
    plt.plot(h_list, query_time, label='x-h,y-query time')
    # plt.legend()
    # plt.show()
    ax1 = plt.subplot(143)
    query_time = [0.04633188247680664, 0.07162914276123047, 0.11865706443786621]
    plt.plot(h_list, query_time, label='x-h,y-query time')

    ax1 = plt.subplot(144)
    recall=[0.840,0.8966,0.9800]
    plt.plot(h_list, recall, label='x-h,y-recall')
    # plt.legend()

    precision=[0.6012915975,0.86432317,0.99397540]
    plt.plot(h_list, precision, label='x-h,y-precision')
    # plt.legend()
    plt.legend()
    plt.show()
    r_list = [2, 3, 4, 5, 10]
    into_bucket_time=[0.9411962032318115,0.7856254577636719,0.5645124912261963,0.5663774013519287,0.49138784408569336]


if __name__ == '__main__':
    draw()
