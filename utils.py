from numpy import hsplit, sum, ndarray, unique
from numpy.random import default_rng


def calc_moments(b1, b2, moment, rms, gradient, t):
    v = b1 * moment + (1 - b1) * gradient
    s = b2 * rms + (1 - b2) * gradient ** 2
    return v, s, v / (1 - b1 ** t), s / (1 - b2 ** t)


def sep(data: ndarray):
    return hsplit(data, [-1])


def max_len(arr: ndarray):
    return 0 if arr.shape[0] == 0 else max(len(v) for v in unique(arr))


def avg(data: ndarray, axis=None):
    return sum(data, axis=axis) / len(data)


def if_dropout(data: ndarray, rate, dropout):
    if dropout and rate < 1:
        mask = default_rng().random(data.shape) < rate
        return data * mask / (1 - rate)
    return data


def mini_batch(func):
    def wrapper(self, data: ndarray, labels: ndarray, *args, **kwargs):
        m = data.shape[0]
        samples = default_rng().choice(m, replace=False, size=(min(64, m)))
        return func(self, data[samples], labels[samples], *args, **kwargs)
    return wrapper
