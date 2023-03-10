from numpy import ndarray, unique, sum as np_sum, log2, nonzero, delete, argmax, arange, full, array, concatenate
from numpy.random import choice
from utils import max_len
from concurrent.futures import ProcessPoolExecutor


class RandomForest:

    def __init__(self):
        self.trees = [DecisionTree() for _ in range(64)]
        self.bags = None

    def train(self, data: ndarray):
        m, p = data.shape
        p -= 1
        self.bags = [concatenate((choice(p, round(p ** 0.5), replace=False), arange(1) - 1)) for _ in range(len(self.trees))]
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(picklable_train, tree, data[choice(m, m)][:, bag]) for i, (tree, bag) in enumerate(zip(self.trees, self.bags))]
        self.trees = [v.result() for v in futures]

    def predict(self, data: ndarray):
        predictions = array([tree.predict(data[:, bag]) for tree, bag in zip(self.trees, self.bags)])
        return unique(predictions.T, axis=1)[:, 0]


class DecisionTree:

    def __init__(self):
        self.subtrees = None
        self.split = None, None, 0
        self.label = None

    def info_gain(self, data: ndarray, mask: ndarray):
        left, right = data[mask], delete(data, mask, axis=0)
        n_l, n_r = left.shape[0], right.shape[0]
        return (n_l * self.entropy(left.T[-1]) + n_r * self.entropy(right.T[-1])) / (n_l + n_r)

    def set_best_split(self, data: ndarray):
        for i, feat in enumerate(data.T[:-1]):
            for v in unique(feat)[1:]:
                self.split = max(self.split, (i, v, self.info_gain(data, self.get_mask(feat, v))), key=lambda a: a[-1])

    def train(self, data: ndarray):
        self.set_best_split(data)
        if self.split != (None, None, 0):
            self.subtrees = [DecisionTree(), DecisionTree()]
            mask = self.get_mask(data)
            left, right = data[mask], delete(data, mask, axis=0)
            for i, v in enumerate((left, right)):
                self.subtrees[i].train(v)
        else:
            labels, counts = unique(data[:, -1], return_counts=True)
            self.label = labels[argmax(counts)]

    def entropy(self, labels: ndarray):
        p = self.get_prob(labels)
        return np_sum(-p * log2(p + 1e-9))

    def predict(self, data: ndarray):
        if self.split == (None, None, 0):
            return full(data.shape[0], self.label)
        l_mask = self.get_mask(data)
        r_mask = delete(arange(data.shape[0]), l_mask)
        left, right = [self.subtrees[i].predict(data[v]) for i, v in enumerate((l_mask, r_mask))]
        results = ndarray(data.shape[0], dtype=f"S{max(max_len(left), max_len(right))}")
        results[l_mask] = left
        results[r_mask] = right
        return results.astype("str")

    def get_mask(self, data: ndarray, val=None):
        if val is None:
            data, val = data[:, self.split[0]], self.split[1]
        return nonzero(data == val)

    def __repr__(self):
        return f"({self.label} {self.split}" + f" left: {self.subtrees[0]} right: {self.subtrees[1]})" if self.subtrees else ""

    @staticmethod
    def get_prob(data: ndarray):
        return unique(data, return_counts=True)[1] / data.shape[0]


def picklable_train(tree, data: ndarray):
    tree.train(data)
    return tree
