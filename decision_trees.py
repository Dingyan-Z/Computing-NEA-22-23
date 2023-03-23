from numpy import ndarray, unique, sum as np_sum, log2, nonzero, delete, argmax, arange, full, array, concatenate
from numpy.random import choice
from utils import max_len
from concurrent.futures import ProcessPoolExecutor
from sys import stdout


class RandomForest:

    def __init__(self):
        self.trees = [DecisionTree() for _ in range(128)]
        self.bags = None

    def train(self, data: ndarray):  # trains all the trees
        n = len(self.trees)
        m, p = data.shape
        p -= 1
        self.bags = [concatenate((choice(p, round(p ** 0.5), replace=False), arange(1) - 1)) for _ in range(n)]  # randomly selecting subsets of data
        with ProcessPoolExecutor() as executor:  # parallel processing
            futures = [executor.submit(picklable_train, tree, data[choice(m, m)][:, bag]) for i, (tree, bag) in enumerate(zip(self.trees, self.bags))]  # submitting threads
            count = 0
            print("\nProgress")
            while count < n:  # gather threads whilst displaying progress
                new_count = [v.done() for v in futures].count(True)
                if new_count != count:
                    stdout.write("\b" * n)
                    stdout.write(f"{new_count}/{n}")
                    count = new_count
            print()
            self.trees = [v.result() for v in futures]  # trained trees

    def predict(self, data: ndarray):  # predicts labels for input data
        predictions = array([tree.predict(data[:, bag]) for tree, bag in zip(self.trees, self.bags)])
        return unique(predictions.T, axis=1)[:, 0]


class DecisionTree:

    def __init__(self):
        self.subtrees = None
        self.split = None, None, 0
        self.label = None

    def info_gain(self, data: ndarray, mask: ndarray):  # vectorised calculation for information gian
        left, right = data[mask], delete(data, mask, axis=0)  # split to left, right branch based on condition
        n_l, n_r = left.shape[0], right.shape[0]
        return (n_l * self.entropy(left.T[-1]) + n_r * self.entropy(right.T[-1])) / (n_l + n_r)  # formula

    def set_best_split(self, data: ndarray):  # given data, find the best way to split it
        for i, feat in enumerate(data.T[:-1]):  # iterate through all possible splits
            for v in unique(feat)[1:]:
                self.split = max(self.split, (i, v, self.info_gain(data, self.get_mask(feat, v))), key=lambda a: a[-1])

    def train(self, data: ndarray):  # recursively trains a tree, creating branches and leaf nodes
        self.set_best_split(data)
        if self.split != (None, None, 0):  # if there is a distinct best split
            self.subtrees = [DecisionTree(), DecisionTree()]  # left and right branch
            mask = self.get_mask(data)
            left, right = data[mask], delete(data, mask, axis=0)
            for i, v in enumerate((left, right)):  # recursively set best split for children nodes
                self.subtrees[i].train(v)
        else:
            labels, counts = unique(data[:, -1], return_counts=True)
            self.label = labels[argmax(counts)]  # getting most common label

    def entropy(self, labels: ndarray):  # vectorised entropy calculations
        p = self.get_prob(labels)
        return np_sum(-p * log2(p + 1e-9))

    def predict(self, data: ndarray):  # predicts labels for datas
        if self.split == (None, None, 0):  # if empty split then this tree is leaf node
            return full(data.shape[0], self.label)
        l_mask = self.get_mask(data)
        r_mask = delete(arange(data.shape[0]), l_mask)
        left, right = [self.subtrees[i].predict(data[v]) for i, v in enumerate((l_mask, r_mask))]  # splits data based on condition
        results = ndarray(data.shape[0], dtype=f"S{max(max_len(left), max_len(right))}")  # Numpy arrays are immutable so length of each string must be precalculated
        results[l_mask] = left  # merge results using indices
        results[r_mask] = right
        return results.astype("str")

    def get_mask(self, data: ndarray, val=None):  # applies condition to every element and returns binary output in a same sized matrix
        if val is None:
            data, val = data[:, self.split[0]], self.split[1]
        return nonzero(data == val)

    def __repr__(self):
        return f"({self.label} {self.split}" + f" left: {self.subtrees[0]} right: {self.subtrees[1]})" if self.subtrees else ""

    @staticmethod
    def get_prob(data: ndarray):  # returns expected probability of elements
        return unique(data, return_counts=True)[1] / data.shape[0]


def picklable_train(tree, data: ndarray):  # a stateless decision tree trainer
    tree.train(data)
    return tree
