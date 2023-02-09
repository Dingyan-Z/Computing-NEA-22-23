from neural_net import Dense
from csv import reader
from numpy import split, array, atleast_1d, ndarray, nonzero
from matplotlib.pyplot import ylabel, plot, legend, show, xlabel, title
from utils import sep
from reinforcement_learning import RL
from decision_trees import DecisionTree, RandomForest


with open("abalone.csv", "r") as file:
    data = array(list(reader(file))[1:])[:100]


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
        rl.train()
    net = rl.get_net()
    print(net)
    return net


def dt_test(training_data: ndarray, test_data: ndarray, rf=False):
    model = RandomForest if rf else DecisionTree()
    model.train(training_data)
    feats, labels = sep(test_data)
    results = model.predict(feats)
    # print(nonzero(results == labels.T[0])[0].shape[0] / results.shape[0] * 100, "%")


if __name__ == '__main__':
    split_data = split(data, [int(len(data) * 0.8)])
    rf_test(*split_data)
    # nn_test(*split_data, rl_test(*split_data))
