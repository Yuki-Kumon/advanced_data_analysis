# -*- coding: utf-8 -*-

"""
宿題2。スパースなガウスモデル。
Author :
    Yuki Kumon
Last Update :
    2019-04-27
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_sample(xmin, xmax, sample_size, true_f):
    '''
    ノイズを加えた学習データを作成する
    '''
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    target = eval(true_f)
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    '''
    計画行列を作成する
    '''
    # x[None] - c[:, None]の部分で、計画行列の引数の組のところを作っている(このとき、2次元配列になっている)
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


def minimum_square_error(indexes, y1, y2):
    '''
    最小二乗誤差を計算する
    インデックスを用いて計算する
    '''
    error = 0.0
    for i in indexes:
        error += (y1[i] - y2[i])**2 / 2
    return error


def cross_validation(x, y, c, h, lam, k):
    '''
    cross validationにより推定の妥当性を計算する
    '''
    error_average = 0.0
    for i in range(int(len(x) / k)):
        # テスト用データ
        '''
        x_test = x[k * i:k * (i + 1):1]
        y_test = y[k * i:k * (i + 1):1]
        '''
        # 学習用データ
        if i == 0:
            x_train = x[k * (i + 1)::1]
            y_train = y[k * (i + 1)::1]
        else:
            x_train = np.r_[x[:k * i:1], x[k * (i + 1)::1]]
            y_train = np.r_[y[:k * i:1], y[k * (i + 1)::1]]
        # 学習用データを用いて、計画行列を計算する
        kappa_train = calc_design_matrix(x_train, x_train, h)
        # 学習した結果から全領域のデータを書き出すための計画行列
        kappa = calc_design_matrix(x_train, x, h)
        # 交互方向乗数法により反復計算で解を探す
    # print(error_average)
    return error_average


def loop(x, y, kappa, lam, loop_max):
    '''
    交互方向乗数法により、スパースな解を見つける
    '''
    # initialize
    theta = np.empty_like(x)
    z = theta.copy()
    u = np.empty_like(x)

    # update
    for i in range(loop_max):
        theta = np.linalg.inv((kappa.T.dot(kappa) + np.eye(len(x)))).dot(kappa.T.dot(y) + z - u)
        z = np.maximum(0, theta + u - lam) + np.minimum(0, theta + u + lam)
        u = u + theta - z
    # drop minor thetas
    theta = np.where(theta < 0.001, 0.0, theta)
    return theta, z, u


if __name__ == '__main__':
    np.random.seed(0)  # set the random seed for reproducibility
    
    # set parameters
    sample_size = 50
    xmin, xmax = -3, 3
    lam = 0.001
    h = 1.0

    # create sample
    true_f = 'np.sin(np.pi * x) / (np.pi * x) + 0.1 * x'  # 正解の関数
    x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size, true_f=true_f)

    # create design matrix
    kappa = calc_design_matrix(x, x, h)

    # calc parameters
    theta, z, u = loop(x, y, kappa, lam, 10000)
    # print(theta)
    # count zero thetas
    print('zero parameters quantity: ', str(np.sum(theta < 0.001)))

    # create data to visualize the prediction
    X = np.linspace(start=xmin, stop=xmax, num=5000)
    k = calc_design_matrix(x, x, h)
    K = calc_design_matrix(x, X, h)
    theta = np.linalg.solve(
        k.T.dot(k) + lam * np.identity(len(k)),
        k.T.dot(y[:, None]))
    prediction = K.dot(theta)
    # visualization
    plt.clf()
    plt.scatter(x, y, c='green', marker='o')
    plt.plot(X, prediction)
    plt.savefig('Lec3_homework2_result.png')
