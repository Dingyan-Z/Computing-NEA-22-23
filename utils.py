import numpy as np


def safe_log(n):
    return 0 if n <= 0 else np.log(n)


def calc_moments(b1, b2, moment, rms, gradient, t):
    v = b1 * moment + (1 - b1) * gradient
    s = b2 * rms + (1 - b2) * gradient ** 2
    return v, s, v / (1 - b1 ** t), s / (1 - b2 ** t)
