from numpy import array, unique, sum as np_sum, ndarray, zeros, zeros_like
from activation_funcs import LeakyReLU, Tanh
from neural_net import Dense
from time import perf_counter
from utils import avg, sep


class RL:

    def __init__(self, training_data: ndarray, test_data: ndarray, ratio=0.2, alpha=0.01, gamma=0.8):
        self.alpha = alpha
        self.gamma = gamma
        self.training_data = sep(training_data)
        self.test_data = sep(test_data)
        output_layer = Tanh if unique(test_data[-1]).shape[0] == 2 else LeakyReLU
        self.net = Dense([self.training_data[0].shape[1], 2, 1], [LeakyReLU] * 2 + [output_layer])
        self.weights = zeros(5)
        self.prev_q, self.prev_feats = self.evaluate(self.net)
        self.prev_reward = 0
        self.ratio = ratio

    def update(self, reward):
        difference = reward - self.prev_q + self.gamma * max(self.evaluate(next_net)[0] for next_net in self.get_actions())
        self.weights *= difference

    def train(self):
        self.net = self.choose_action()
        self.update(self.get_reward(self.net))

    def get_actions(self):
        shared_range = self.net.get_changeable_layers()
        actions = [self.net, self.net.add_layer((max(shared_range) + 1) if shared_range else 1, LeakyReLU)]
        for layer in shared_range:
            actions += [self.net.add_node(layer), self.net.pop_node(layer), self.net.pop_layer(layer), self.net.add_layer(layer, LeakyReLU)]
        return filter(lambda a: a is not None, actions)

    def choose_action(self):
        actions = self.get_actions()
        (self.prev_q, self.prev_feats), action = max([(self.evaluate(v), v) for v in actions], key=lambda a: a[0][0])
        return action

    def get_reward(self, net: Dense):
        start = perf_counter()
        reward = self.test_net(net.copy()) / (perf_counter() - start) ** self.ratio
        self.prev_reward, reward = reward, self.prev_reward - reward
        return reward

    def test_net(self, net):
        for _ in range(50000):
            net.train(*self.training_data)
        return net.cost(*self.test_data)

    def evaluate(self, net: Dense):
        feats = self.get_feats(net)
        return feats.dot(self.weights), feats

    def get_net(self):
        return self.net.copy()

    def get_feats(self, net: Dense):
        layer_sizes, units = net.get_info()
        feats = zeros_like(self.weights)
        feats[0] = len(layer_sizes)
        feats[1] = avg(layer_sizes)
        differences = array([layer_sizes[i] - layer_sizes[i + 1] for i in range(len(layer_sizes) - 1)])
        feats[2] = np_sum(differences)
        feats[3] = np_sum(differences ** 2)
        feats[4] = np_sum(layer_sizes)
        return feats / 10000
