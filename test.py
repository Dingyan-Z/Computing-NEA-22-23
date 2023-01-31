from neural_net import Dense
from csv import reader
from numpy import split, array, atleast_1d, ndarray
from matplotlib.pyplot import ylabel, plot, legend, show, xlabel, title
from utils import sep
from reinforcement_learning import RL
from Runtime import get_runtime


with open("abalone.csv", "r") as file:
    data = array(list(reader(file))[1:], dtype=float)

# with open("concrete_data.csv", "r") as file:
#     data = array(list(reader(file))[1:], dtype=float)

split_data = split(data, [int(len(data) * 0.8)])


@get_runtime
def nn_test(training_data: ndarray, test_data: ndarray, net: Dense):
    epochs = 100000
    nn = atleast_1d(net)
    title("NN Test Data")
    xlabel("Epochs")
    ylabel("Cost")
    for i, v in enumerate(nn):
        cost_history = []
        for _ in range(epochs):
            v.train(*sep(training_data))
            cost_history.append(v.cost(*sep(test_data)))
        plot(range(len(cost_history)), cost_history, label=i)
    legend()
    show()


def rl_test(training_data: ndarray, test_data: ndarray):
    epochs = 100
    rl = RL(training_data, test_data)
    for i in range(epochs):
        print(i)
        rl.train()
    net = rl.get_net()
    print(net)
    return net


if __name__ == '__main__':
    nn_test(*split_data, rl_test(*split_data))
