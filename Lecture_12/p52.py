import numpy as np
import matplotlib
import scipy.linalg

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)


def data_generation(n=1000):
    a = 3. * np.pi * np.random.rand(n)
    x = np.stack(
        [a * np.cos(a), 30. * np.random.random(n), a * np.sin(a)], axis=1)
    return a, x


def LapEig(x, d=10):
    # implement here
    # 類似度行列を計算
    """
    W = np.ones([len(x), len(x)])
    for i in range(len(x)):
        for j in range(len(x)):
            if (abs(i - j) > d):
                W[i, j] = 0.
    """
    W = np.zeros([len(x), len(x)])
    for i in range(len(x)):
        # i番目の座標との距離によってソート
        dist = np.empty(len(x))
        for ix in range(len(x)):
            dist[ix] = np.linalg.norm(x[i] - x[ix])
        index = np.argsort(dist)[:d]
        W[i, index] = np.ones(d)
    # ラプラス固有写像により変換
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    # 一般化固有値問題を解く
    eig_val, eig_vec = scipy.linalg.eig(L, D)
    # print(L)
    # print(eig_vec)
    # 変換は0を除く最小の固有値に対応する固有ベクトルで行う
    # pass
    # print(L.dot(np.ones(len(L))))
    return eig_vec[:, 1:]


def visualize(x, z, a):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=a, marker='o')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(z[:, 1], z[:, 0], c=a, marker='o')
    fig.savefig('lecture10-h2.png')


n = 1000
a, x = data_generation(n)
z = LapEig(x)
# print(z)
visualize(x, z, a)
