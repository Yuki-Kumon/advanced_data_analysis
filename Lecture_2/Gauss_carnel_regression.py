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

    @staticmethod
    def minimum_square_error(indexes, y1, y2):
        '''
        最小二乗誤差を計算する
        インデックスを用いて計算する
        '''
        error = 0.0
        for i in indexes:
            error += (y1[i] - y2[i])**2 / 2
        return error

    @classmethod
    def cross_validation(self, x, c, h, lam, k):
        '''
        cross validationにより推定の妥当性を計算する
        '''
        error_all = 0.0
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
            kappa_train = self.calc_design_matrix(x_train, x_train, h)
            # 学習した結果から全領域のデータを書き出すための計画行列
            kappa = self.calc_design_matrix(x_train, x, h)
            # 最小二乗誤差の計算によりパラメータを推定(解析解があるので連立方程式を解く)
            theta = np.linalg.solve(
                kappa_train.T.dot(kappa_train) + lam * np.identity(len(kappa_train)),
                kappa_train.T.dot(y_train[:, None]))
            # 予想データを出力
            prediction = kappa.dot(theta)
            # テストデータを用いて妥当性を検証する
            error = self.minimum_square_error(np.array(range(k * i, k * (i + 1))), prediction[:, 0], y)
            error_all += error / float(int(len(x) / k))
        print(error_all)
        return error_all




if __name__ == '__main__':
    """
    # a[行指定, 列指定]！
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
    h = 0.1
    lam = 0.3
    k = 3
    Gauss_carnel_regression.cross_validation(x, x, h, lam, k)
