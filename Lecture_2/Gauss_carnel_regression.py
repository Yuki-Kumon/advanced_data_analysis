# -*- coding: utf-8 -*-

"""
宿題1。ガウスカーネルモデルのl2-正則化を用いた最小二乗回帰の実装。
Author :
    Yuki Kumon
Last Update :
    2019-04-16
"""

import numpy as np
import matplotlib


class Gauss_carnel_regression:
    '''
    ガウスカーネルモデルの計算に用いる関数群をまとめたクラス
    '''
    @staticmethod
    def generate_sample(xmin, xmax, sample_size, true_f):
        '''
        ノイズを加えた学習データを作成する
        '''
        x = np.linspace(start=xmin, stop=xmax, num=sample_size)
        target = eval(true_f)
        noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
        return x, target + noise

    @staticmethod
    def calc_design_matrix(x, c, h):
        '''
        計画行列を作成する
        '''
        # x[None] - c[:, None]の部分で、計画行列の引数の組のところを作っている(このとき、2次元配列になっている)
        return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


if __name__ == '__main__':
    """
    # 注意。a[行指定, 列指定]です！
    test = np.array([[11, 12], [21, 22]])
    print(test)
    print(test[:, 0])

    test1 = np.array([0, 1, 2])
    print(test1[None] - test1[:, None])
    """

    np.random.seed(0)  # set the random seed for reproducibility

    # create sample
    sample_size = 50
    xmin, xmax = -3, 3
    true_f = 'np.sin(np.pi * x) / (np.pi * x) + 0.1 * x'  # 正解の関数
    x, y = Gauss_carnel_regression.generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size, true_f=true_f)
