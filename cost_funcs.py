import numpy as np
import math


def linear(predictions: np.ndarray, labels: np.ndarray):
    return sum([(prediction - label) ** 2 for prediction, label in zip(predictions, labels)]) / 2 / len(predictions)


def binary_cross_entropy(predictions: np.ndarray, labels: np.ndarray):
    return sum([-label * math.log(prediction) - (1 - label) * math.log(1 - prediction) for prediction, label in zip(predictions, labels)]) / len(predictions)
