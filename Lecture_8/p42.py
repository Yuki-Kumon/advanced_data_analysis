import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)  # set the random seed for reproducibility


def data_generate(n=50):
    x = np.random.randn(n, 3)
    x[:n // 2, 0] -= 15
    x[n // 2:, 0] -= 5
    x[1:3, 0] += 10
    x[:, 2] = 1
    y = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
    index = np.random.permutation(np.arange(n))
    return x[index], y[index]


def predict(train_x, train_y, gamma):
    # initialize
    mu = np.matrix(np.zeros(3)).T
    sigma = np.arange(9).reshape([3, 3])

    # predict
    for i in range(len(train_x)):
        x_here = np.matrix(train_x[i]).T
        # print(mu)
        # print((train_y[i] * np.maximum(np.matrix([0]), 1 - mu.T.dot(x_here) * train_y[i])[0, 0] * sigma.dot(x_here)))
        sigma_temp = sigma - sigma.dot(x_here).dot(x_here.T).dot(sigma) / (x_here.T.dot(sigma).dot(x_here) + gamma)[0, 0]
        mu_temp = mu + (train_y[i] * np.maximum(np.matrix([0]), 1 - mu.T.dot(x_here) * train_y[i])[0, 0] * sigma.dot(x_here))[:, 0] / (x_here.T.dot(sigma).dot(x_here) + gamma)[0, 0]
        sigma = sigma_temp
        mu = mu_temp
    return np.array(mu.T)[0]


n, N = 50, 1000
h = 0.3
lr = 0.1

x, y = data_generate()
theta = predict(x, y, 0.1)
print(theta)

# print(x[y == -1][:, 0])

# visualize
plt.clf()
# plt.figure(figsize=(6, 6))
# plt.xlim(-5., 5.)
# plt.ylim(-7., 7.)
lin = np.array([-16., -4.])
plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1],
            marker='$O$', c='blue')
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
            marker='$X$', c='red')
plt.savefig('lecture8.png')
