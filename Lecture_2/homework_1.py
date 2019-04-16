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
    def generate_sample(xmin, xmax, sample_size):
        '''
        ノイズを加えた学習データを作成する
        '''
        x = np.linspace(start=xmin, stop=xmax, num=sample_size)
        pix = np.pi * x
        target = np.sin(pix) / pix + 0.1 * x
        noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
        return x, target + noise

    @staticmethod
    def calc_design_matrix(x, c, h):
        '''
        計画行列を作成する
        '''
        return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


if __name__ == '__main__':
    test = range(10)
