# -*- coding: utf-8 -*-

"""
宿題3。テューキー回帰の繰り返し最小二乗アルゴリズムの実装。
Author :
    Yuki Kumon
Last Update :
    2019-05-08
"""


import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)


def generate_sample(x_min=-3., x_max=3., sample_size=10):
    x = np.linspace(x_min, x_max, num=sample_size)
    y = x + np.random.normal(loc=0., scale=.2, size=sample_size)
    y[-1] = y[-2] = y[1] = -4  # outliers
    return x, y


def build_design_matrix(x):
    # 基底関数はtheta_1とtheta_2 x の2つ
    # 計画行列は(xの次元)×2の大きさになる
    phi = np.empty(x.shape + (2,))
    phi[:, 0] = 1.
    phi[:, 1] = x
    return phi


def loop(x, y, phi, theta, eta):
    '''
    繰り返し最小二乗アルゴリズムのループ部分
    '''
    # 解との偏差のベクトルr~を計算する
    r_tilde = phi.dot(theta) - y
    # 重みベクトルw~を計算する
    w_tilde = np.where((r_tilde < eta) & (-eta < r_tilde), (1 - r_tilde**2 / eta**2)**2, 0)
    # w~により計算される対角行列W~を計算する
    W_tilde = np.diag(w_tilde)
    # thetaを重み付き最小二乗回帰の解析解を用いて更新する
    return np.linalg.inv(phi.T.dot(W_tilde.dot(phi))).dot(phi.T.dot(W_tilde).dot(y))


def iterative_reweighted_least_squares(x, y, eta=1., n_iter=1000):
    # create design matrix
    phi = build_design_matrix(x)
    # initialize theta
    # 基底関数が2つなのでパラメータは2つ
    theta = np.empty(2)
    # ループ計算により解を漸近させる
    for _ in range(n_iter):
        theta_new = loop(x, y, phi, theta, eta)
        if np.linalg.norm(theta - theta_new) < 1e-3:
            theta = theta_new
            break
        else:
            theta = theta_new
    return theta


if __name__ == '__main__':
    # create dataset
    x, y = generate_sample()
    theta = iterative_reweighted_least_squares(x, y)
