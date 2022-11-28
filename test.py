import neural_net
import activation_funcs as activation
import csv
import numpy as np

nn = neural_net.Dense([7, 3, 1], [activation.LeakyReLU, activation.LeakyReLU, activation.LeakyReLU])
with open("abalone.csv", "r") as file:
    data = np.array(list(csv.reader(file))[1:], dtype=float)
training_data, test_data = np.split(data, [int(len(data) * 0.8)])
nn.train(*np.hsplit(training_data, [-1]))
print(len(training_data), len(data))

