import numpy as np
import random


def safe_log(n):
    return 0 if n <= 0 else np.log(n)


def calc_moments(b1, b2, moment, rms, gradient, t):
    v = b1 * moment + (1 - b1) * gradient
    s = b2 * rms + (1 - b2) * gradient ** 2
    return v, s, v / (1 - b1 ** t), s / (1 - b2 ** t)


def sep(data: np.ndarray):
    return np.hsplit(data, [-1])


def avg(data: np.ndarray, axis=None):
    return np.sum(data, axis=axis) / len(data)


def if_dropout(data: np.ndarray, rate, dropout):
    if dropout and rate < 1:
        for datum in data:
            for i, v in enumerate(datum):
                if random.random() > rate:
                    datum[i] = 0
        return data / (1 - rate)
    return data


def mini_batch(func):
    def wrapper(self, data: np.ndarray, labels: np.ndarray, *args, **kwargs):
        samples = np.random.randint(0, len(data), size=min(64, len(data)))
        return func(self, data[samples], labels[samples], *args, **kwargs)
    return wrapper
