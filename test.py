import neural_net
import activation_funcs as activation
import csv
import numpy as np
import matplotlib.pyplot as plt
import utils

epochs = 10000
nn = neural_net.Dense([7, 3, 1], [activation.LeakyReLU, activation.LeakyReLU])
with open("abalone.csv", "r") as file:
    data = np.array(list(csv.reader(file))[1:], dtype=float)
training_data, test_data = np.split(data, [int(len(data) * 0.8)])
cost_history = []
for _ in range(epochs):
    nn.train(*utils.sep(training_data))
    cost_history.append(nn.cost(*utils.sep(test_data)))
plt.title("Cost")
plt.plot(range(epochs), cost_history)
plt.show()
