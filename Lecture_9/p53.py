import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2.
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = 0
    y[n_positive:] = 1
    return x, y


def cwls(train_x, train_y, test_x):
    # 最適化のためscipyをimportしておく
    from scipy.optimize import minimize

    # Aとbを計算する関数を定義しておく
    def A(x, y, y1, y2):
        index1 = np.where(y == y1)[0]
        index2 = np.where(y == y2)[0]
        A_y1y2 = 0.0
        for i in index1.tolist():
            for j in index2.tolist():
                A_y1y2 += np.linalg.norm(x[i] - x[j])
        A_y1y2 = A_y1y2 / len(index1) / len(index2)

        return A_y1y2

    def b(x1, y, x2, y1):
        index1 = np.where(y == y1)[0]
        b_y1 = 0.0
        for i in index1.tolist():
            for j in range(len(x2)):
                b_y1 += np.linalg.norm(x1[i] - x2[j])
        b_y1 = b_y1 / len(index1) / len(x2)
        return b_y1

    # 計画行列の計算を行う関数を定義
    def design_mat(X):
        '''
        計画行列。サイズは(データ数) * 3
        基底関数: x[0], x[1], 1
        '''
        phi = np.hstack((X, np.ones([len(X), 1])))
        return phi

    # train_yを変換しておく
    train_y = np.where(train_y == 0, -1, 1)
    # print(train_x)
    # J(pi)を最小にするpiを計算する
    pi_tilde = (A(train_x, train_y, 1, -1) - A(train_x, train_y, -1, -1) - b(train_x, train_y, test_x, 1) + b(train_x, train_y, test_x, -1))
    pi_tilde = pi_tilde / (2 * A(train_x, train_y, 1, -1) - A(train_x, train_y, 1, 1) - A(train_x, train_y, -1, -1))
    pi_tilde = min(1, max(0, pi_tilde))
    # 計画行列を計算
    phi = design_mat(train_x)

    # scipyで最適化を行う関数を定義
    def objective_function(theta):
        loss = 0.0
        pis = np.where(train_y == 1, pi_tilde, 1 - pi_tilde)
        # print(pis)
        for i in range(len(train_x)):
            loss += pis[i] * (phi[i].dot(theta.T) - train_y[i])**2
        return loss
    # 最小化
    theta0 = [10.0, 10.0, 10.0]
    res = minimize(objective_function, theta0).x
    # print(minimize(objective_function, theta0))

    print('pi_tilde: {}'.format(pi_tilde))

    return res


def visualize(train_x, train_y, test_x, test_y, theta):
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.xlim(-5., 5.)
        plt.ylim(-7., 7.)
        lin = np.array([-5., 5.])
        plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
        plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1],
                    marker='$O$', c='blue')
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
                    marker='$X$', c='red')
        plt.savefig('lecture9-h3-{}.png'.format(name))


train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta)
