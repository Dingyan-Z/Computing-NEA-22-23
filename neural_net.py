import activation_funcs
import numpy as np


class Dense:

    def __init__(self, layer_sizes, units):
        self.layers = [DenseLayer(layer_activations[i], size, layer_sizes[i - 1] if i > 0 else 1, inp_layer=i == 0) for i, size in enumerate(layer_sizes)]
        self.num_layers = len(layer_sizes)

    def predict(self, data: np.ndarray):
        output = [data]
        for layer in self.layers:
            output.append(layer.predict(output[-1]))
        return output

    def train(self, data: np.ndarray, labels: np.ndarray):
        for (datum, label) in zip(data, labels):
            datum = np.atleast_2d(datum)
            predictions = self.predict(datum)
            activations, results = predictions[:-1], predictions[-1]
            deltas = (results - label) * self.layers[-1].gradient(results)
            # self.layers[-1].update(deltas, results, len(datum))
            for activation, layer in zip(reversed(activations), reversed(self.layers[:-1])):
                print(deltas, layer.gradient(activation))
                deltas = layer.predict(deltas, reverse=True) * layer.gradient(activation)
                # layer.update(deltas, activation, len(datum))


class DenseLayer:

    def __init__(self, activation, num_units, num_feats, inp_layer=False):
        self.units = [activation(num_feats) for _ in range(num_units)]
        self.inp_layer = inp_layer

    def gradient(self, data: np.ndarray):
        return self.units[0].gradient(data)

    def predict(self, data: np.ndarray, reverse=False):
        return np.array([unit.predict(data[:, i] if self.inp_layer else data, reverse) for i, unit in enumerate(self.units)]).T

    def update(self, deltas, activation, m):
        for unit in self.units:
            unit.update(deltas, activation, m)
