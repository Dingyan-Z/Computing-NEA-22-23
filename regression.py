import numpy as np
import utils
import matplotlib.pyplot as plt
import math


class Regression:

    def __init__(self, num_features, alpha=0.1, lambda_reg=0.005, b1=0.9, b2=0.999, eps=1e-8):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.bias = 0
        self.iterations = 0
        self.v_dw = np.zeros(num_features)
        self.s_dw = np.zeros(num_features)
        self.v_db = self.s_db = 0
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def predict(self, data):
        raise NotImplementedError

    def cost(self, data: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    def train(self, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, epochs, graph=False):
        costs = []
        weights_history = []
        bias_history = []
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2, nrows=2)
        fig.tight_layout(pad=2)
        ax0.set_title("Predictions")
        ax1.set_title("Cost")
        ax2.set_title("Weights")
        ax3.set_title("Bias")
        data_range = range(math.floor(np.amin(test_data)), math.ceil(np.amax(test_data)) + 1)
        if graph:
            ax0.plot(data_range, self.predict(np.array(data_range)))
        for _ in range(epochs):
            mini_batch = (train_data, train_labels)
            if len(train_data) > 32:
                samples = np.random.randint(0, len(train_data), size=32)
                mini_batch = (train_data[samples], train_labels[samples])
            self.update(*mini_batch)
            if graph:
                costs.append(self.cost(test_data, test_labels))
                weights_history.append(sum(self.weights))
                bias_history.append(self.bias)
        if graph:
            ax0.plot(data_range, self.predict(np.array(data_range)))
            ax1.plot(range(epochs), costs)
            ax2.plot(range(epochs), weights_history)
            ax3.plot(range(epochs), bias_history)
            ax0.plot(test_data, test_labels, "ro")
            ax0.legend(["Before", "After", "Data"])
            plt.show()

    def update(self, data: np.ndarray, labels: np.ndarray):
        self.iterations += 1
        m = len(data)
        differences = self.predict(data) - labels
        for i in range(self.num_features):
            dw = self.alpha * (differences.dot(data[:, i]) + self.lambda_reg * (self.weights[i])) / m
            self.v_dw[i], self.s_dw[i], v_dw_cor, s_dw_cor = utils.calc_moments(self.b1, self.b2, self.v_dw[i], self.s_dw[i], dw, self.iterations)
            self.weights[i] -= self.alpha * v_dw_cor / (s_dw_cor ** 0.5 + self.eps)
        db = self.alpha * sum(differences) / m
        self.v_db, self.s_db, v_db_cor, s_db_cor = utils.calc_moments(self.b1, self.b2, self.v_db, self.s_db, db, self.iterations)
        self.bias -= self.alpha * v_db_cor / (s_db_cor ** 0.5 + self.eps)


class Linear(Regression):

    def predict(self, data):
        return np.array([self.weights.dot(features) + self.bias for features in data])

    def cost(self, data: np.ndarray, labels: np.ndarray):
        predictions = self.predict(data)
        return (sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels)) + self.lambda_reg * sum(np.square(self.weights))) / 2 / len(predictions)


class Logistic(Regression):

    def predict(self, data):
        return np.array([1 / (1 + np.exp(-(self.weights.dot(features) + self.bias))) for features in data])

    def cost(self, data: np.ndarray, labels: np.ndarray):
        predictions = self.predict(data)
        return (sum(-label * utils.safe_log(prediction) - (1 - label) * utils.safe_log(1 - prediction) for prediction, label in zip(predictions, labels)) + self.lambda_reg / 2 * sum(np.square(self.weights))) / len(predictions)
