import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n=200):
    x = np.linspace(0, np.pi, n // 2)
    u = np.stack([np.cos(x) + .5, -np.sin(x)], axis=1) * 10.
    u += np.random.normal(size=u.shape)
    v = np.stack([np.cos(x) - .5, np.sin(x)], axis=1) * 10.
    v += np.random.normal(size=v.shape)
    x = np.concatenate([u, v], axis=0)
    y = np.zeros(n)
    y[0] = 1
    y[-1] = -1
    return x, y


def calc_design_matrix(x, c, h):
    '''
    計画行列を作成する
    '''
    # x[None] - c[:, None]の部分で、計画行列の引数の組のところを作っている(このとき、2次元配列になっている)
    # return np.sum(np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2)), axis=2)
    phi = np.zeros([len(c), len(x)])
    for i in range(len(c)):
        for j in range(len(x)):
            phi[i, j] = np.exp(- np.linalg.norm(c[i] - x[j])**2 / 2 / h**2)
    return phi


def lrls(x, y, h=1., l=1., nu=1.):
    """

    :param x: data points
    :param y: labels of data points
    :param h: width parameter of the Gaussian kernel
    :param l: weight decay
    :param nu: Laplace regularization
    :return:
    """

    #
    # Implement this function
    #

    # p15の式をそのまま実装
    # print(x)
    # print(y)
    x1 = np.array([x[0], x[-1]])
    x = np.concatenate([x1, x[1:-1]])
    phi = calc_design_matrix(x, x, h)
    phi_hat = calc_design_matrix(x, x[:2], h)
    # print(phi_hat)
    # print(phi.shape)
    # print(np.sum(phi, axis=1))
    L = np.diag(np.sum(phi, axis=1)) - phi
    # print(L)
    # print(phi_hat)
    return np.linalg.inv(phi_hat.T.dot(phi_hat) + l * np.eye(len(x)) + 2 * nu * phi.T.dot(L).dot(phi)).dot(phi_hat.T).dot(np.array([y[0], y[-1]]))


def visualize(x, y, theta, h=1.):
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xlim(-20., 20.)
    plt.ylim(-20., 20.)
    grid_size = 100
    grid = np.linspace(-20., 20., grid_size)
    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
    k = np.exp(-np.sum((x.astype(np.float32)[:, None] - mesh_grid.astype(
        np.float32)[None]) ** 2, axis=2).astype(np.float64) / (2 * h ** 2))
    plt.contourf(X, Y, np.reshape(np.sign(k.T.dot(theta)),
                                  (grid_size, grid_size)),
                 alpha=.4, cmap=plt.cm.coolwarm)
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker='$.$', c='black')
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='$X$', c='red')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker='$O$', c='blue')
    plt.savefig('lecture9-h1.png')


x, y = generate_data(n=200)
theta = lrls(x, y, h=1.)
# print(theta)
visualize(x, y, theta)
