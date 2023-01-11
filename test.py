import neural_net
import activation_funcs as activation
import csv
import numpy as np
import matplotlib.pyplot as plt
import utils
import reinforcement_learning
import Runtime


with open("abalone.csv", "r") as file:
    data = np.array(list(csv.reader(file))[1:], dtype=float)
split_data = np.split(data, [int(len(data) * 0.8)])


@Runtime.get_runtime
def nn_test(training_data: np.ndarray, test_data: np.ndarray):
    epochs = 10000
    nn = (neural_net.Dense([7, 4, 2, 1], [activation.LeakyReLU] * 3),
          neural_net.Dense([7, 2, 1], [activation.LeakyReLU] * 2))
    plt.title("NN Test Data")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    for i, v in enumerate(nn):
        cost_history = []
        for _ in range(epochs):
            v.train(*utils.sep(training_data))
            cost_history.append(v.cost(*utils.sep(test_data)))
        plt.plot(range(len(cost_history)), cost_history, label=i)
    plt.legend()
    plt.show()


def rl_test(training_data: np.ndarray, test_data: np.ndarray):
    epochs = 20
    rl = reinforcement_learning.RL(training_data, test_data)
    for i in range(epochs):
        if i % 10 == 0:
            print(i)
            print(rl.get_net())
        rl.train()
    print(rl.get_net())


nn_test(*split_data)
# rl_test(*split_data)

