# -*- coding: utf-8 -*-

"""
宿題2。ガウスカーネルモデルを用いた手書きの数字の識別。
学習したパラメータを元にラベルを予測。
Author :
    Yuki Kumon
Last Update :
    2019-05-17
"""


from number_image_classify import *
import numpy as np


def predict(train, test, theta, bandwidth):
    phi = build_design_mat(train, test, bandwidth)
    return phi.T.dot(theta)


# 学習済みパラメータの読み込み
width = 10.0
thetas = np.load('thetas.npy')
xs = np.load('xs.npy')
# データローダーの準備
csv_root = '/Users/yuki_kumon/Documents/python/advanced_data_analysis/Lecture_5/digit/'
loader = dataloader(csv_root)

# 予測(0について)
test_img = loader('test', 0)
out = np.empty([10, len(test_img)])
for num in range(10):
    train_img = xs[num]
    theta = thetas[num]
    out[num] = predict(train_img, test_img, theta, width)
prediction = np.argmax(out, axis=0)
print(prediction)
