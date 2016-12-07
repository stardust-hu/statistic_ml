# coding: utf8
"""
@Author: yuhao
@Email: yuhao.hu1992@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def sigmoid(x):
    y = 2 / (1 + np.exp(-x)) - 1
    return np.round(y)


class Perceptron(object):
    def __init__(self, step=1, max_iteration=100):
        self.step = step
        self.max_iteration = max_iteration

    def train(self, X, labels):
        n_samples, n_features = X.shape
        if type(labels) is np.ndarray:
            self.n_label = list(set(labels.tolist()))
        else:
            self.n_label = list(set(labels))
        Y = map(lambda x: -1 if x == self.n_label[0] else 1, labels)
        self.w = np.zeros((n_features, 1))
        self.b = 0
        for i in range(self.max_iteration):
            for sample in range(n_samples):
                x = X[sample]
                y = Y[sample]
                loss = y * (np.matrix(x) * np.matrix(self.w) + self.b)
                if loss[0, 0] <= 0:
                    self.w += self.step * y * x.reshape(n_features, 1)
                    self.b += self.step * y

    def predict(self, X):
        y = sigmoid(np.matrix(X) * np.matrix(self.w) + self.b)
        y_predict = map(lambda x: self.n_label[0] if x[0] == -1 else self.n_label[1], y)
        return y_predict


def plot_data(data, labels, w=None, b=None, score=None):
    if data.shape[1] > 2:
        print u'数据超过２维！'
        return False
    else:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels, s=50)

        if (w is not None) and (b is not None) and (score is not None):
            x1 = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
            x2 = -1 * (w[0] * x1 + b) / w[1]  # w1*x1 + w2*x2 + b = 0
            plt.hold(1)
            plt.plot(x1, x2)
            plt.title('score: %f%%' % (score*100))
        plt.show()
        return True


def test():
    n_data = 30
    X1 = np.random.randn(n_data, 2) + 0 * np.ones((n_data, 2))
    X2 = np.random.randn(n_data, 2) + 2 * np.ones((n_data, 2))
    X = np.vstack((X1, X2))
    labels = [0] * n_data + [1] * n_data
    plot_data(X, labels)
    p = Perceptron()
    p.train(X, labels)
    plot_data(X, labels, p.w, p.b)
    y_predict = p.predict(X)
    score = accuracy_score(labels, y_predict)
    print u'准确率: %f' % score
    plot_data(X, y_predict, p.w, p.b, score)


def test2():
    n_data = 30
    X1 = np.random.randn(n_data, 2) + 0 * np.ones((n_data, 2))
    X2 = np.random.randn(n_data, 2) + 2 * np.ones((n_data, 2))
    X = np.vstack((X1, X2))
    labels = [0] * n_data + [1] * n_data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
    p = Perceptron()
    p.train(X_train, y_train)
    y_predict = p.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    print score


if __name__ == '__main__':
    test()
    # test2()
