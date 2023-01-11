import numpy as np
import activation_funcs
import neural_net
import time
import utils


class RL:

    def __init__(self, training_data: np.ndarray, test_data: np.ndarray, alpha=0.01, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.training_data = utils.sep(training_data)
        self.test_data = utils.sep(test_data)
        self.net = neural_net.Dense([self.training_data[0].shape[1], 2, 1], [activation_funcs.LeakyReLU] * 3)
        self.weights = np.zeros(4)
        self.prev_q, self.prev_feats = self.evaluate(self.net)
        self.prev_reward = 0

    def update(self, reward):
        difference = reward - self.prev_q + self.gamma * max(self.evaluate(next_net)[0] for next_net in self.get_actions())
        self.weights += difference * self.alpha * self.prev_feats

    def train(self):
        self.net = self.choose_action()
        self.update(self.get_reward(self.net))

    def get_actions(self):
        shared_range = self.net.get_changeable_layers()
        return filter(lambda a: a is not None, [net for layer in shared_range for net in (self.net.add_node(layer), self.net.pop_node(layer), self.net.pop_layer(layer), self.net.add_layer(layer, activation_funcs.LeakyReLU, 2))] + [self.net.add_layer((max(shared_range) + 1) if shared_range else 1, activation_funcs.LeakyReLU, 2), self.net])

    def choose_action(self):
        # print([(self.evaluate(v), v) for v in self.get_actions()])
        (q, feats), action = max([(self.evaluate(v), v) for v in self.get_actions()], key=lambda a: a[0][0])
        self.prev_q, self.prev_feats = q, feats
        return action

    def get_reward(self, net: neural_net.Dense):
        start = time.perf_counter()
        reward = utils.avg([self.test_net(net.copy()) for _ in range(1)]) / (time.perf_counter() - start) ** 0.5
        self.prev_reward, reward = reward, self.prev_reward - reward
        return reward

    def test_net(self, net):
        for _ in range(1000):
            net.train(*self.training_data)
        return net.cost(*self.test_data)

    def evaluate(self, net: neural_net.Dense):
        feats = self.get_feats(net)
        return feats.dot(self.weights), feats

    def get_net(self):
        return self.net.copy()

    def get_feats(self, net: neural_net.Dense):
        layer_sizes, units = net.get_info()
        feats = np.zeros_like(self.weights)
        feats[0] = len(layer_sizes)
        feats[1] = utils.avg(layer_sizes)
        feats[2] = sum([(layer_sizes[i] - layer_sizes[i + 1]) ** 2 for i in range(len(layer_sizes) - 1)])
        feats[3] = sum(layer_sizes)
        return feats / 1000
