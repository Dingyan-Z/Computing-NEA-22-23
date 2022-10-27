import numpy as np


def linear(weights: np.ndarray, bias, predictions: np.ndarray, data: np.ndarray, labels: np.ndarray, alpha):
    m = len(weights)
    for i in range(m):
        weights[i] -= alpha * sum([features[i] * (prediction - label) for features, prediction, label in zip(data, predictions, labels)]) / m
    bias -= alpha * sum([prediction - label for prediction, label in zip(predictions, labels)]) / m
    return weights, bias
