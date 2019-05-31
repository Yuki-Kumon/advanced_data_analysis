# -*- coding: utf-8 -*-

"""
宿題1。ガウスカーネルモデルに対する最小二乗確率的分類。
Author :
    Yuki Kumon
Last Update :
    2019-05-31
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(sample_size, n_class):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y


def visualize(x, y, theta, h):
    X = np.linspace(-5., 5., num=1000)
    K = np.exp(-(x - X[:, None]) ** 2 / (2 * h ** 2))

    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-.3, 1.8)
    logit = K.dot(theta)
    """
    unnormalized_prob = np.exp(logit - np.max(logit, axis=1, keepdims=True))
    prob = unnormalized_prob / unnormalized_prob.sum(1, keepdims=True)
    """
    # 解の補正
    prob = np.maximum(logit, 0) / np.sum(np.maximum(logit, 0), axis=1, keepdims=True)

    plt.plot(X, prob[:, 0], c='blue')
    plt.plot(X, prob[:, 1], c='red')
    plt.plot(X, prob[:, 2], c='green')

    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')

    plt.savefig('lecture7-p17.png')


def learning(x, y, lam, h, n_class):
    sample_size = len(x)
    theta = np.random.normal(size=(sample_size, n_class))
    phi = build_design_mat(x, x, h)
    for i in range(n_class):
        theta[:, i] = np.linalg.inv(phi.T.dot(phi) + lam * np.eye(len(phi))).dot(phi.T).dot(
            np.where(y == i, 1., 0.)
        )
    return theta


def build_design_mat(x, c, h):
    '''
    ガウスカーネルモデルの計画行列を計算する
    '''
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


if __name__ == '__main__':
    x, y = generate_data(sample_size=90, n_class=3)
    # print(x)
    theta = learning(x, y, h=2., lam=.1, n_class=3)
    visualize(x, y, theta, h=2.)
