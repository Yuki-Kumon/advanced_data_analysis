import numpy as np
import matplotlib
from scipy.linalg import eig

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(46)


def generate_data(sample_size=100, pattern='two_cluster'):
    if pattern not in ['two_cluster', 'three_cluster']:
        raise ValueError('Dataset pattern must be one of '
                         '[two_cluster, three_cluster].')
    x = np.random.normal(size=(sample_size, 2))
    if pattern == 'two_cluster':
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def fda(x, y):
    """Fisher Discriminant Analysis.
    Implement this function

    Returns
    -------
    T : (1, 2) ndarray
        The embedding matrix.
    """

    import scipy.linalg

    classes = np.unique(y)
    # クラス間散布行列、クラス内散布行列の計算
    Sb = np.zeros([len(x[0]), len(x[0])])
    Sw = np.zeros_like(Sb)
    for i, class_num in enumerate(classes):
        class_index = np.where(y == class_num)
        mean_vec = np.matrix(np.mean(x[class_index], axis=0))
        Sb += len(class_index[0]) * mean_vec.T.dot(mean_vec)
        Sw += (np.matrix(x[class_index[0]]) - mean_vec).T.dot((np.matrix(x[class_index[0]]) - mean_vec))

    """
    C = (np.matrix(x)).T.dot((np.matrix(x)))
    print(C)
    print(Sw + Sb)
    """
    eig_val, eig_vec = scipy.linalg.eig(Sb, Sw)  # 一般固化有値問題を解く
    return eig_vec[:1, :] / np.linalg.norm(eig_vec[:1, :])


def visualize(x, y, T):
    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    plt.plot(np.array([-T[:, 0], T[:, 0]]) * 9,
             np.array([-T[:, 1], T[:, 1]]) * 9, 'k-')
    plt.legend()
    plt.savefig('lecture11-h1.png')


sample_size = 100
x, y = generate_data(sample_size=sample_size, pattern='two_cluster')
# x, y = generate_data(sample_size=sample_size, pattern='three_cluster')
T = fda(x, y)
visualize(x, y, T)
