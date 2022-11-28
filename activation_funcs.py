import numpy as np
import utils
import random


class Base:

    def __init__(self, num_feats, alpha=0.1, lambda_reg=0.005, b1=0.9, b2=0.999, eps=1e-8):
        self.weights = np.zeros(num_feats)
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.bias = random.random()
        self.iterations = 0
        self.v_dw = np.zeros(num_feats)
        self.s_dw = np.zeros(num_feats)
        self.v_db = self.s_db = 0
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def predict(self, data: np.ndarray, reverse=False):
        raise NotImplementedError

    def gradient(self, data: np.ndarray):
        raise NotImplementedError

    def update(self, deltas, activation, m):
        self.iterations += 1
        for i, weight in enumerate(self.weights):
            dw = (sum(deltas * activation[:, i]) + self.lambda_reg * weight) / m
            self.v_dw[i], self.s_dw[i], v_dw_cor, s_dw_cor = utils.calc_moments(self.b1, self.b2, self.v_dw[i], self.s_dw[i], dw, self.iterations)
            self.weights[i] -= self.alpha * v_dw_cor / (s_dw_cor ** 0.5 + self.eps)
        db = sum(deltas) / m
        self.v_db, self.s_db, v_db_cor, s_db_cor = utils.calc_moments(self.b1, self.b2, self.v_db, self.s_db, db, self.iterations)
        self.bias -= self.alpha * v_db_cor / (s_db_cor ** 0.5 + self.eps)

    def get_weights(self):
        return self.weights


class Tanh(Base):

    @utils.auto_reshape
    def predict(self, data: np.ndarray, reverse=False):
        matrices = data, self.weights
        return np.tanh(np.dot(*(matrices[::-1] if reverse else matrices)))

    def gradient(self, data: np.ndarray):
        return 1 - np.tanh(data) ** 2


class LeakyReLU(Base):

    @utils.auto_reshape
    def predict(self, data: np.ndarray, reverse=False):
        matrices = data, self.weights
        if reverse:
            return np.array([sum(datum * weight for weight in self.weights) for datum in data])
        z = np.dot(*(matrices[::-1] if reverse else matrices))
        return np.maximum(self.alpha * z, z)

    def gradient(self, data: np.ndarray):
        return np.where(data > 0, 1, self.alpha)
