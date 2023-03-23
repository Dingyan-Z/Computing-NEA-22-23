from numpy import tanh, maximum, where, ndarray
from numpy.random import default_rng


class Tanh:

    @staticmethod
    def predict(z: ndarray, *_):  # Tanh formula
        return tanh(z)

    @staticmethod
    def gradient(data: ndarray, *_):  # calculates gradient
        return 1 - tanh(data) ** 2

    @staticmethod
    def init_weights(inputs, units):  # sampling from normal distribution to initialise weights
        bound = 6 ** 0.5 / (inputs + units) ** 0.5
        return default_rng().uniform(-bound, bound, (inputs, units))


class LeakyReLU:

    @staticmethod
    def predict(z: ndarray, alpha):  # LeakyReLU formula
        return maximum(alpha * z, z)

    @staticmethod
    def gradient(data: ndarray, alpha):  # calculates gradient
        return where(data > 0, 1, alpha)

    @staticmethod
    def init_weights(inputs, units):  # Xavier initialisation
        return default_rng().normal(0, (2 / inputs) ** 0.5, (inputs, units))
