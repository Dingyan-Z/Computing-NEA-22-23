import numpy as np
import math


class Regression:

    def __init__(self, num_features, alpha):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.alpha = alpha
        self.bias = 0

    def predict(self, data):
        raise NotImplementedError

    def cost(self, data: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    def train(self, data: np.ndarray, labels: np.ndarray):
        raise NotImplementedError


class Linear(Regression):

    def predict(self, data):
        return np.ndarray([self.weights.dot(features) + self.bias for features in data])

    def cost(self, data: np.ndarray, labels: np.ndarray):
        predictions = self.predict(data)
        return sum([(prediction - label) ** 2 for prediction, label in zip(predictions, labels)]) / 2 / len(predictions)

    def train(self, data: np.ndarray, labels: np.ndarray):
        m = len(data)
        predictions = self.predict(data)
        for i in range(self.num_features):
            self.weights[i] -= self.alpha * sum([features[i] * (prediction - label) for features, prediction, label in zip(data, predictions, labels)]) / m
        self.bias -= self.alpha * sum([prediction - label for prediction, label in zip(predictions, labels)]) / m


class Logistic(Regression):

    def predict(self, data):
        return np.ndarray([1 / (1 + math.e ** -(self.weights.dot(features) + self.bias)) for features in data])

    def cost(self, data: np.ndarray, labels: np.ndarray):
        predictions = self.predict(data)
        return sum([-label * math.log(prediction) - (1 - label) * math.log(1 - prediction) for prediction, label in zip(predictions, labels)]) / len(predictions)

    def train(self, data: np.ndarray, labels: np.ndarray):
        m = len(data)
        predictions = self.predict(data)
        for i in range(self.num_features):
            self.weights[i] -= self.alpha * sum([features[i] * (prediction - label) for features, prediction, label in zip(data, predictions, labels)]) / m
        self.bias -= self.alpha * sum([prediction - label for prediction, label in zip(predictions, labels)]) / m


if __name__ == '__main__':
    print()
