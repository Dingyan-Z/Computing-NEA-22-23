import numpy as np
import utils


class Regression:

    def __init__(self, num_features, alpha=0.1, b1=0.9, b2=0.999, eps=1e-8):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.alpha = alpha
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

    def train(self, data: np.ndarray, labels: np.ndarray):
        self.iterations += 1
        m = len(data)
        differences = self.predict(data) - labels
        for i in range(self.num_features):
            dw = self.alpha * differences.dot(data[:, i]) / m
            self.v_dw[i] = (self.b1 * self.v_dw[i] + (1 - self.b1) * dw)
            self.s_dw[i] = (self.b2 * self.s_dw[i] + (1 - self.b2) * dw ** 2)
            v_dw_cor = self.v_dw[i] / (1 - self.b1 ** self.iterations)
            s_dw_cor = self.s_dw[i] / (1 - self.b2 ** self.iterations)
            self.weights[i] -= self.alpha * v_dw_cor / (s_dw_cor ** 0.5 + self.eps)
        db = self.alpha * sum(differences) / m
        self.v_db = (self.b1 * self.v_db + (1 - self.b1) * db)
        self.s_db = (self.b2 * self.s_db + (1 - self.b2) * db ** 2)
        v_db_cor = self.v_db / (1 - self.b1 ** self.iterations)
        s_db_cor = self.s_db / (1 - self.b2 ** self.iterations)
        self.bias -= self.alpha * v_db_cor / (s_db_cor ** 0.5 + self.eps)


class Linear(Regression):

    def predict(self, data):
        return np.array([self.weights.dot(features) + self.bias for features in data])

    def cost(self, data: np.ndarray, labels: np.ndarray):
        predictions = self.predict(data)
        return sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels)) / 2 / len(predictions)


class Logistic(Regression):

    def predict(self, data):
        return np.array([1 / (1 + np.exp(-(self.weights.dot(features) + self.bias))) for features in data])

    def cost(self, data: np.ndarray, labels: np.ndarray):
        predictions = self.predict(data)
        print(predictions)
        return sum(-label * utils.safe_log(prediction) - (1 - label) * utils.safe_log(1 - prediction) for prediction, label in zip(predictions, labels)) / len(predictions)
