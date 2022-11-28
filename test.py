import neural_net
import activation_funcs as activation
import csv
import numpy as np
import matplotlib.pyplot as plt

epochs = 1000
nn = neural_net.Dense([7, 3, 1], [activation.LeakyReLU, activation.LeakyReLU, activation.LeakyReLU])
with open("abalone.csv", "r") as file:
    data = np.array(list(csv.reader(file))[1:], dtype=float)
training_data, test_data = np.split(data, [int(len(data) * 0.8)])
cost_history = []
for _ in range(epochs):
    nn.train(*np.hsplit(training_data, [-1]))
    cost_history.append(nn.cost(*np.hsplit(test_data, [-1])))
plt.title("Cost")
plt.plot(range(epochs), cost_history)
plt.show()
