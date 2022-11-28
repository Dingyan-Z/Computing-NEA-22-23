import numpy as np
import utils


class Dense:

    def __init__(self, layer_sizes, units, alpha=0.001):
        self.layers = [np.random.rand(layer_sizes[i], v) for i, v in enumerate(layer_sizes[1:])]
        self.bias = 0
        self.units = units
        self.alpha = alpha

    def predict(self, data: np.ndarray):
        output = [data]
        for unit, weights in zip(self.units, self.layers):
            output.append(unit.predict(output[-1].dot(weights), self.alpha))
        output[-1] + self.bias
        return output

    @utils.mini_batch
    def train(self, data: np.ndarray, labels: np.ndarray):
        for datum, label in zip(data, labels):
            predictions = self.predict(np.atleast_2d(datum))
            deltas = np.sum(predictions[-1] - label) * self.units[-1].gradient(predictions[-1], self.alpha)
            for i, (activation, weights) in enumerate(zip(reversed(predictions[:-1]), reversed(self.layers))):
                self.layers[-i - 1] -= self.alpha * activation.T.dot(deltas)
                deltas = deltas[-1].dot(weights.T) * self.units[i].gradient(activation, self.alpha)

    def cost(self, data: np.ndarray, labels: np.ndarray):
        return np.sum((self.predict(data)[-1] - labels) ** 2) / len(data)
