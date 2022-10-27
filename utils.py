import numpy as np
import math


def linear(weights: np.ndarray, bias, data: np.ndarray):
    return [features.dot(weights) + bias for features in data]


def logistic(weights: np.ndarray, bias, data: np.ndarray):
    return [1 / (1 + math.e ** -(weights.dot(features) + bias)) for features in data]

