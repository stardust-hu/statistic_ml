# coding: utf8
"""
@Author: yuhao
@Email: yuhao.hu1992@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
style.use('ggplot')


class Perceptron(object):
    def __init__(self, step=0.001, max_iteration=1000):
        self.step = step
        self.max_iteration = max_iteration

    def train(self, features, labels):
        pass

    def predict(self, features):
        return 0


def plot_data(data, labels):
    if data.shape[1] > 2:
        print u'数据超过２维！'
        return False
    else:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels, s=50)
        plt.show()
        return True


def test():
    n_data = 50
    X1 = np.random.randn(n_data, 2) + 0 * np.ones((n_data, 2))
    X2 = np.random.randn(n_data, 2) + 5 * np.ones((n_data, 2))
    X = np.vstack((X1, X2))
    # print X
    labels = [0] * n_data + [1] * n_data
    plot_data(X, labels)
    p = Perceptron()
    p.train(X, labels)
    y_predict = p.predict(X)


if __name__ == '__main__':
    test()
