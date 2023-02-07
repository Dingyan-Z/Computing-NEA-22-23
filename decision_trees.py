from numpy import ndarray, unique, sum as np_sum, log2, nonzero, delete, str_, argmax, arange, zeros, full


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
        if labels.dtype.type is str_:
            p = self.get_prob(labels)
            return np_sum(-p * log2(p + 1e-9))
        return labels.var()

    def predict(self, data: ndarray):
        if self.split == (None, None, 0):
            return full(data.shape[0], self.label)
        l_mask = self.get_mask(data)
        r_mask = delete(arange(data.shape[0]), l_mask)
        left, right = [self.subtrees[i].predict(data[v]) for i, v in enumerate((l_mask, r_mask))]
        results = zeros(data.shape[0], dtype=object)
        results[l_mask] = left
        results[r_mask] = right
        return results

    def get_mask(self, data: ndarray, val=None):
        if val is None:
            data, val = data[:, self.split[0]], self.split[1]
        return nonzero(data == val if data.dtype.type is str_ else data < val)

    def __repr__(self):
        return f"({self.label} {self.split}" + f" left: {self.subtrees[0]} right: {self.subtrees[1]})" if self.subtrees else ""

    @staticmethod
    def get_prob(data: ndarray):
        return unique(data, return_counts=True)[1] / data.shape[0]

