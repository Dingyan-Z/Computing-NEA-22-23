from numpy import tanh, maximum, where, ndarray
from numpy.random import default_rng


class Tanh:

    @staticmethod
    def predict(z: ndarray, *_):
        return tanh(z)

    @staticmethod
    def gradient(data: ndarray, *_):
        return 1 - tanh(data) ** 2

    @staticmethod
    def init_weights(inputs, units):
        bound = 6 ** 0.5 / (inputs + units) ** 0.5
        return default_rng().uniform(-bound, bound, (inputs, units))


class LeakyReLU:

    @staticmethod
    def predict(z: ndarray, alpha):
        return maximum(alpha * z, z)

    @staticmethod
    def gradient(data: ndarray, alpha):
        return where(data > 0, 1, alpha)

    @staticmethod
    def init_weights(inputs, units):
        return default_rng().normal(0, (2 / inputs) ** 0.5, (inputs, units))
