from numpy import zeros, ndarray, dot, exp, log, square, sum as np_sum, round, abs as np_abs
from utils import mini_batch, calc_moments


class Base:

    def __init__(self, num_features, alpha=0.1, lambda_=0.005, b1=0.9, b2=0.999, eps=1e-8):
        self.num_features = num_features
        self.weights = zeros(num_features)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.bias = 0
        self.iterations = 0
        self.v_dw = zeros(num_features)
        self.s_dw = zeros(num_features)
        self.v_db = self.s_db = 0
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def predict(self, data: ndarray):  # return predicted labels for data
        raise NotImplementedError

    def cost(self, data: ndarray, labels: ndarray):  # return cost at current epoch
        raise NotImplementedError

    @mini_batch
    def train(self, data: ndarray, labels: ndarray):  # trains model for 1 epoch
        self.iterations += 1
        m = len(data)
        differences = self.predict(data) - labels.T
        for i in range(self.num_features):
            dw = (differences.dot(data[:, i]) + np_sum(np_abs(self.lambda_ * self.weights))) / m  # gradient
            self.v_dw[i], self.s_dw[i], v_dw_cor, s_dw_cor = calc_moments(self.b1, self.b2, self.v_dw[i], self.s_dw[i], dw, self.iterations)
            self.weights[i] -= self.alpha * v_dw_cor / (s_dw_cor ** 0.5 + self.eps)  # ADAM
        db = np_sum(differences) / m
        self.v_db, self.s_db, v_db_cor, s_db_cor = calc_moments(self.b1, self.b2, self.v_db, self.s_db, db, self.iterations)
        self.bias -= self.alpha * v_db_cor / (s_db_cor ** 0.5 + self.eps)


class Linear(Base):

    def predict(self, data: ndarray, rounded=True):
        return dot(data, self.weights) + self.bias

    def cost(self, data: ndarray, labels: ndarray):
        predictions = self.predict(data, False)
        return (np_sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels)) + self.lambda_ * np_sum(square(self.weights))) / 2 / len(predictions)


class Logistic(Base):

    def predict(self, data: ndarray, rounded=True):
        predictions = 1 / (1 + exp(-dot(data, self.weights) - self.bias))
        return round(predictions).astype(bool) if rounded else predictions

    def cost(self, data: ndarray, labels: ndarray):
        predictions = self.predict(data, False)
        return (np_sum(-label * log(prediction + 1e-9) - (1 - label) * log(1 - prediction + 1e-9) for prediction, label in zip(predictions, labels)) + self.lambda_ / 2 * np_sum(square(self.weights))) / len(predictions)
