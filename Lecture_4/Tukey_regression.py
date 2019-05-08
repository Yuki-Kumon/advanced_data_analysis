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


def predict(x, theta):
    phi = build_design_matrix(x)
    return phi.dot(theta)


def loop(x, y, phi, theta, eta):
    '''
    繰り返し最小二乗アルゴリズムのループ部分
    '''
    # 解との偏差のベクトルr~を計算する
    r_tilde = phi.dot(theta) - y
    # 重みベクトルw~を計算する(r_tildeの要素を置き換える形で)
    np.where((r_tilde < eta) and (-eta < r_tilde), (1 - r_tilde**2 / eta**2)**2, 0)


def iterative_reweighted_least_squares(x, y, eta=1., n_iter=1000):
    # create design matrix
    phi = build_design_matrix(x)
    # initialize theta
    theta = np.empty_like(x)


if __name__ == '__main__':
    x = np.empty(10)
    print(build_design_matrix(x))
