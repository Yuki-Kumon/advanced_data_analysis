import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def data_generation1(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, np.random.randn(n, 1)],
                          axis=1)


def data_generation2(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, 2 * np.round(
        np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.], axis=1)


def calc_design_matrix(x, c, h):
    '''
    計画行列を作成する
    '''
    # x[None] - c[:, None]の部分で、計画行列の引数の組のところを作っている(このとき、2次元配列になっている)
    # return np.sum(np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2)), axis=2)
    phi = np.zeros([len(c), len(x)])
    for i in range(len(c)):
        for j in range(len(x)):
            phi[i, j] = np.exp(- np.linalg.norm(c[i] - x[j])**2 / 2 / h**2)
    return phi


def cal_lpp(x, t):
    '''
    局所性保存射影の変換行列を計算する
    '''
    # 近傍の重み行列を計算
    W = calc_design_matrix(x, x, t)

    # 対角行列Dを計算
    D = np.diag(np.diag(W))

    # 一般化固有値問題を解く
    A = x.T.dot((D - W)).dot(x)
    B = x.T.dot(D).dot(x)
    print(B)

    return 0


n = 100
n_components = 1
# x = data_generation1(n)
x = data_generation2(n)

hoge = cal_lpp(x, 1.0)

plt.xlim(-6., 6.)
plt.ylim(-6., 6.)
plt.plot(x[:, 0], x[:, 1], 'rx')
# plt.plot(np.array([-v[:, 0], v[:, 0]]) * 9, np.array([-v[:, 1], v[:, 1]]) * 9)
plt.savefig('lecture10-p44.png')
