# -*- coding: utf-8 -*-

"""
宿題2。ガウスカーネルモデルを用いた手書きの数字の識別。
1対多法を用いる。
Author :
    Yuki Kumon
Last Update :
    2019-05-08
"""


import numpy as np
import csv
import os
import glob


def build_design_mat(x1, x2, bandwidth):
    '''
    ガウスカーネルモデルの計画行列を計算する
    '''
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))


def load_csv(path):
    '''
    csvファイルから数字データを取得。numpy配列に格納する。
    '''
    # 先に行数を取得
    length = sum(1 for line in open(path))
    with open(path) as f:
        reader = csv.reader(f)
        images = np.empty([length, 256])
        num = 0
        for row in reader:
            image = np.array(row)
            # image = image_row.reshape([16, 16])
            images[num] = image
            num += 1
    return images


def get_csv_path_list(root):
    '''
    データセットフォルダのcsvのパスを取得。正規表現でtrainとtestを分類しておく。
    '''
    train_list = glob.glob(os.path.join(root, '*train*.csv'))
    test_list = glob.glob(os.path.join(root, '*test*.csv'))
    return train_list, test_list


class dataloader():
    '''
    データローダー
    '''

    def __init__(self, csv_root):
        # csvロード用のパス
        train, test = get_csv_path_list(csv_root)
        # ファイル名のみのリストを作成
        train_name = []
        test_name = []
        for name in train:
            train_name.append(os.path.splitext(os.path.basename(name))[0])
        for name in test:
            test_name.append(os.path.splitext(os.path.basename(name))[0])

        # csvファイルから画像をまとめて読み込んでおく
        images_dict = {}
        for i in range(len(train)):
            path = train[i]
            digit = train_name[i]
            images = load_csv(path)
            images_dict[digit] = images
        for i in range(len(test)):
            path = test[i]
            digit = test_name[i]
            images = load_csv(path)
            images_dict[digit] = images
        self.images_dict = images_dict

    """
    def __call__(self, mode, dig_num, idx):
        digit_name = 'digit_' + mode + str(dig_num)
        images = self.images_dict[digit_name]
        return images[idx]
    """
    def __call__(self, mode, dig_num):
        '''
        画像のセットをそのまま返すようにした
        '''
        digit_name = 'digit_' + mode + str(dig_num)
        images = self.images_dict[digit_name]
        return images


def train(digit):
    '''
    一対多法の学習関数
    '''
    # まず画像ファイルを読み込むための数字のリストを作成。
    nums = range(10)
    del nums[digit]


if __name__ == '__main__':
    csv_path = '/Users/yuki_kumon/Documents/python/advanced_data_analysis/Lecture_5/digit/digit_test2.csv'
    csv_root = '/Users/yuki_kumon/Documents/python/advanced_data_analysis/Lecture_5/digit/'
    a = dataloader(csv_root)
