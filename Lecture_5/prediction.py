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

# 全ての番号について、テストデータで検証
confusion_matrix = np.empty([10, 10])
for target in range(10):
    test_img = loader('test', target)
    out = np.empty([10, len(test_img)])
    for num in range(10):
        train_img = xs[num]
        theta = thetas[num]
        out[num] = predict(train_img, test_img, theta, width)
    prediction = np.argmax(out, axis=0)
    # update confusion matrix
    for i in range(10):
        confusion_matrix[target, i] = np.sum(np.where(prediction == i, 1, 0))
print(confusion_matrix)

"""
結果                   出       力
    [[196.   0.   1.   1.   0.   0.   1.   0.   1.   0.]
     [  0. 199.   0.   0.   0.   1.   0.   0.   0.   0.]
入   [  0.   0. 187.   1.   7.   0.   0.   2.   3.   0.]
     [  0.   0.   1. 191.   0.   3.   0.   1.   3.   1.]
力   [  0.   1.   1.   0. 185.   0.   3.   0.   2.   8.]
     [  2.   0.   1.  15.   4. 176.   0.   0.   0.   2.]
     [  2.   0.   2.   0.   1.   2. 192.   0.   1.   0.]
     [  0.   1.   2.   0.   2.   0.   0. 186.   1.   8.]
     [  2.   0.   1.   8.   2.   5.   0.   0. 181.   1.]
     [  0.   0.   0.   0.   2.   0.   0.   3.   3. 192.]]
"""


"""
# 予測(0について)
test_img = loader('test', 0)
out = np.empty([10, len(test_img)])
for num in range(10):
    train_img = xs[num]
    theta = thetas[num]
    out[num] = predict(train_img, test_img, theta, width)
prediction = np.argmax(out, axis=0)
print(prediction)
"""
