import numpy as np


class Regression:

    def __init__(self, model, shape, cost, gradient, alpha):
        self.model = model
        self.weights = np.zeros(shape)
        self.cost = cost
        self.gradient = gradient
        self.alpha = alpha
        self.bias = 0

    def predict(self, datum):
        return self.weights.dot(datum) + self.bias

    def train(self, data: np.ndarray, labels: np.ndarray):
        self.weights, self.bias = self.gradient(self.weights, self.bias, self.predict(data), data, labels)


if __name__ == '__main__':
    regr = Linear_Regression(10)
    print(regr.predict([1] * 10))
