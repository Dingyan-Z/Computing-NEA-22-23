import numpy as np


class Tanh:

    @staticmethod
    def predict(z: np.ndarray, *_):
        return np.tanh(z)

    @staticmethod
    def tanh_gradient(data: np.ndarray, *_):
        return 1 - np.tanh(data) ** 2

    @staticmethod
    def init_weights(inputs, units):
        bound = 6 ** 0.5 / (inputs + units) ** 0.5
        return np.random.uniform(-bound, bound, (inputs, units))


class LeakyReLU:

    @staticmethod
    def predict(z: np.ndarray, alpha):
        return np.maximum(alpha * z, z)

    @staticmethod
    def gradient(data: np.ndarray, alpha):
        return np.where(data > 0, 1, alpha)

    @staticmethod
    def init_weights(inputs, units):
        return np.random.normal(0, (2 / inputs) ** 0.5, (inputs, units))
