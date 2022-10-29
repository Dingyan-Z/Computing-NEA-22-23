import numpy as np
import utils


class Base:

    def __init__(self, num_features, alpha=0.1, lambda_reg=0.005, b1=0.9, b2=0.999, eps=1e-8):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.iterations = 0
        self.v_dw = np.zeros(num_features)
        self.s_dw = np.zeros(num_features)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def predict(self, data: np.ndarray):
        raise NotImplementedError

    def update(self, data: np.ndarray, labels: np.ndarray):
        self.iterations += 1
        m = len(data)
        differences = self.predict(data) - labels
        for i in range(self.num_features):
            dw = (differences.dot(data[:, i]) + self.lambda_reg * self.weights[i]) / m
            self.v_dw[i], self.s_dw[i], v_dw_cor, s_dw_cor = utils.calc_moments(self.b1, self.b2, self.v_dw[i], self.s_dw[i], dw, self.iterations)
            self.weights[i] -= self.alpha * v_dw_cor / (s_dw_cor ** 0.5 + self.eps)


class Linear(Base):

    def predict(self, data: np.ndarray):
        return np.dot(data, self.weights)


class Tanh(Base):

    def predict(self, data: np.ndarray):
        return np.tanh(np.dot(data, self.weights))


class LeakyReLU(Base):

    def predict(self, data: np.ndarray):
        z = np.dot(data, self.weights)
        return np.maximum(0.01 * z, z)
