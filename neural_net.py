from copy import deepcopy
from numpy import atleast_2d, sum, zeros, ndarray
from numpy.random import default_rng
from utils import avg, calc_moments, if_dropout, mini_batch


class Dense:

    def __init__(self, layer_sizes, units, alpha=0.001, b1=0.9, b2=0.999, eps=1e-8, rate=None):
        self.units = units
        self.layer_sizes = layer_sizes
        self.units = units
        self.alpha = alpha
        self.iterations = 0
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.rate = ([0.5] * len(layer_sizes)) if rate is None else rate
        self.v_dw = [zeros((self.layer_sizes[i], v)) for i, v in enumerate(self.layer_sizes[1:])]
        self.s_dw = [deepcopy(v) for v in self.v_dw]
        self.v_db = self.s_db = 0
        self.bias = default_rng().random()
        self.layers = [self.units[i].init_weights(*self.layer_sizes[i:i + 2]) for i in range(len(self.layer_sizes) - 1)]

    def add_node(self, layer):
        layer_sizes, units = self.get_info()
        layer_sizes[layer] += 1
        return Dense(layer_sizes, units)

    def add_layer(self, layer, unit, size=2):
        layer_sizes, units = self.get_info()
        layer_sizes.insert(layer, size)
        units.insert(layer, unit)
        return Dense(layer_sizes, units)

    def pop_layer(self, layer):
        layer_sizes, units = self.get_info()
        layer_sizes.pop(layer)
        units.pop(layer)
        return Dense(layer_sizes, units)

    def pop_node(self, layer):
        layer_sizes, units = self.get_info()
        if layer_sizes[layer] > 1:
            layer_sizes[layer] -= 1
            return Dense(layer_sizes, units)

    def get_changeable_layers(self):
        return range(1, len(self.layer_sizes) - 1)

    def predict(self, data: ndarray, activations=False, dropout=False):
        output = [if_dropout(data, self.rate[0], dropout)]
        for unit, weights, rate in zip(self.units, self.layers, self.rate[1:]):
            output.append(if_dropout(unit.predict(output[-1].dot(weights), self.alpha), rate, dropout))
        return output if activations else output[-1]

    @mini_batch
    def train(self, data: ndarray, labels: ndarray):
        self.iterations += 1
        predictions = self.predict(data, activations=True, dropout=True)
        deltas = avg(predictions[-1] - labels)
        self.v_db, self.s_db, v_db_cor, s_db_cor = calc_moments(self.b1, self.b2, self.v_db, self.s_db, deltas, self.iterations)
        self.bias -= self.alpha * v_db_cor / (s_db_cor ** 0.5 + self.eps)
        deltas *= avg(self.units[-1].gradient(predictions[-1], self.alpha))
        for i, (activation, weights) in enumerate(zip(reversed(predictions[:-1]), reversed(self.layers))):
            cor_i = -i - 1
            self.v_dw[cor_i], self.s_dw[cor_i], v_dw_cor, s_dw_cor = calc_moments(self.b1, self.b2, self.v_dw[cor_i], self.s_dw[cor_i], atleast_2d(avg(activation, axis=0)).T.dot(deltas), self.iterations)
            self.layers[cor_i] -= self.alpha * v_dw_cor / (s_dw_cor ** 0.5 + self.eps)
            deltas = atleast_2d(deltas).dot(weights.T) * avg(self.units[i].gradient(activation, self.alpha))

    def get_info(self):
        return [deepcopy(v) for v in (self.layer_sizes, self.units)]

    def cost(self, data: ndarray, labels: ndarray):
        return sum((self.predict(data)[-1] - labels) ** 2) / len(data) / 2

    def copy(self):
        return Dense(*deepcopy(self.get_info()))

    def __repr__(self):
        return f"\nunits: {self.units}\nlayers: {self.layer_sizes}\n"
