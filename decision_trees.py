from numpy import ndarray, unique, sum as np_sum, log2, argwhere, delete, str_


class DecisionTree:

    def __init__(self):
        self.subtrees = None
        self.split = None, None, 0

    def info_gain(self, data, mask):
        left, right = data[mask], delete(data, mask)
        n_l, n_r = left.shape[0] + right.shape[0]
        return (n_l * self.entropy(left) + n_r * self.entropy(right)) / (n_l + n_r)

    def set_best_split(self, data: ndarray):
        for i, feat in enumerate(data.T):
            for v in unique(feat)[1:]:
                col = data[:, feat]
                self.split = max(self.split, (i, v, self.info_gain(col, self.get_mask(data, i, v))), key=lambda a: a[-1])

    def train(self, data):
        self.set_best_split(data)
        if self.split == (None, None, 0):
            return False
        self.subtrees = DecisionTree(), DecisionTree()
        mask = self.get_mask(data, *self.split[:-1])
        for i, v in enumerate((data[mask], delete(data, mask))):
            if self.subtrees[i].train(v) is False:
                del self.subtrees

    def cost(self, data: ndarray):
        return 1 - np_sum(self.get_prob(data) ** 2)

    def entropy(self, data: ndarray):
        p = self.get_prob(data)
        return np_sum(-p * log2(p + 1e-9))

    @staticmethod
    def get_prob(predictions):
        return unique(predictions, return_counts=True) / predictions.shape[0]

    @staticmethod
    def get_mask(data, feat, val):
        col = data[feat]
        return argwhere(col == val if col[0].dtype.type is str_ else col < val)
