import numpy as np
import utils


class Base:

    def __init__(self, num_features, alpha=0.1, lambda_=0.005, b1=0.9, b2=0.999, eps=1e-8):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.bias = 0
        self.iterations = 0
        self.v_dw = np.zeros(num_features)
        self.s_dw = np.zeros(num_features)
        self.v_db = self.s_db = 0
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def predict(self, data: np.ndarray):
        raise NotImplementedError

    def cost(self, data: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    @utils.mini_batch
    def train(self, data: np.ndarray, labels: np.ndarray):
        self.iterations += 1
        m = len(data)
        differences = self.predict(data) - labels.T
        for i in range(self.num_features):
            dw = (differences.dot(data[:, i]) + self.lambda_ * self.weights[i]) / m
            self.v_dw[i], self.s_dw[i], v_dw_cor, s_dw_cor = utils.calc_moments(self.b1, self.b2, self.v_dw[i], self.s_dw[i], dw, self.iterations)
            self.weights[i] -= self.alpha * v_dw_cor / (s_dw_cor ** 0.5 + self.eps)
        db = np.sum(differences) / m
        self.v_db, self.s_db, v_db_cor, s_db_cor = utils.calc_moments(self.b1, self.b2, self.v_db, self.s_db, db, self.iterations)
        self.bias -= self.alpha * v_db_cor / (s_db_cor ** 0.5 + self.eps)


class Linear(Base):

    def predict(self, data: np.ndarray):
        return np.dot(data, self.weights) + self.bias

    def cost(self, data: np.ndarray, labels: np.ndarray):
        predictions = self.predict(data)
        return (sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels)) + self.lambda_ * sum(np.square(self.weights))) / 2 / len(predictions)


class Logistic(Base):

    def predict(self, data: np.ndarray):
        return 1 / (1 + np.exp(np.dot(data, self.weights) + self.bias))

    def cost(self, data: np.ndarray, labels: np.ndarray):
        predictions = self.predict(data)
        return (sum(-label * utils.safe_log(prediction) - (1 - label) * utils.safe_log(1 - prediction) for prediction, label in zip(predictions, labels)) + self.lambda_ / 2 * sum(np.square(self.weights))) / len(predictions)
