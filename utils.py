import numpy as np


def safe_log(n):
    return 0 if n <= 0 else np.log(n)
