import numpy as np
import utils
import random


class Dense:

    def __init__(self, layer_sizes, units, alpha=0.001, b1=0.9, b2=0.999, eps=1e-8, rate=None):
        self.layers = [units[i].init_weights(*layer_sizes[i:i + 2]) for i in range(len(layer_sizes) - 1)]
        self.units = units
        self.alpha = alpha
        self.bias = np.random.random()
        self.iterations = 0
        self.v_dw = [np.zeros((layer_sizes[i], v)) for i, v in enumerate(layer_sizes[1:])]
        self.s_dw = [np.copy(v) for v in self.v_dw]
        self.v_db = self.s_db = 0
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.rate = ([0.8] + [0.5] * (len(layer_sizes) - 2) + [1]) if rate is None else rate

    def predict(self, data: np.ndarray, dropout=False):
        output = [utils.if_dropout(data, self.rate[0], dropout)]
        for unit, weights, rate in zip(self.units, self.layers, self.rate[1:]):
            output.append(utils.if_dropout(unit.predict(output[-1].dot(weights), self.alpha), rate, dropout))
        return output

    @utils.mini_batch
    def train(self, data: np.ndarray, labels: np.ndarray):
        self.iterations += 1
        predictions = self.predict(data, dropout=True)
        deltas = utils.avg(predictions[-1] - labels)
        self.v_db, self.s_db, v_db_cor, s_db_cor = utils.calc_moments(self.b1, self.b2, self.v_db, self.s_db, deltas, self.iterations)
        self.bias -= self.alpha * v_db_cor / (s_db_cor ** 0.5 + self.eps)
        deltas *= utils.avg(self.units[-1].gradient(predictions[-1], self.alpha))
        for i, (activation, weights) in enumerate(zip(reversed(predictions[:-1]), reversed(self.layers))):
            cor_i = -i - 1
            self.v_dw[cor_i], self.s_dw[cor_i], v_dw_cor, s_dw_cor = utils.calc_moments(self.b1, self.b2, self.v_dw[cor_i], self.s_dw[cor_i], np.atleast_2d(utils.avg(activation, axis=0)).T.dot(deltas), self.iterations)
            self.layers[cor_i] -= self.alpha * v_dw_cor / (s_dw_cor ** 0.5 + self.eps)
            deltas = np.atleast_2d(deltas).dot(weights.T) * utils.avg(self.units[i].gradient(activation, self.alpha))

    def cost(self, data: np.ndarray, labels: np.ndarray):
        return np.sum((self.predict(data)[-1] - labels) ** 2) / len(data) / 2

