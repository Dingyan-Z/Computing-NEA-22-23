import neural_net
import utils
import activation_funcs
from numpy import array, arange
import decision_trees


if __name__ == "__main__":

    # neural_net.py
    nn = neural_net.Dense([7, 2, 1], [activation_funcs.LeakyReLU] * 3)  # initialising
    assert(nn.get_info() == [[7, 2, 1], [activation_funcs.LeakyReLU] * 3])
    assert(nn.get_changeable_layers() == range(1, 2))
    deep_copy = nn.copy()  # deep copy
    assert(deep_copy is not nn)  # different memory location
    assert(deep_copy.get_info() == nn.get_info())  # same architecture
    assert(nn.add_layer(2, activation_funcs.LeakyReLU, 3).get_info() == [[7, 2, 3, 1], [activation_funcs.LeakyReLU] * 4])
    assert(nn.add_node(1).get_info() == [[7, 3, 1], [activation_funcs.LeakyReLU] * 3])
    assert(nn.pop_layer(1).get_info() == [[7, 1], [activation_funcs.LeakyReLU] * 2])
    assert(nn.pop_node(1).get_info() == [[7, 1, 1], [activation_funcs.LeakyReLU] * 3])

    # utils.py
    assert(utils.calc_moments(0.9, 0.999, 2, 3, 4, 5) == (2.2, 3.013, 5.372274181338675, 603.8064058024804))  # formula
    assert((utils.sep(array(((1, 2, 3), (1, 2, 3))))[0] == array([[1, 2], [1, 2]])).all())  # separate last column from data
    assert(utils.max_len(array(["a", "ab", "aab", "b"])) == 3)  # longest str
    assert(utils.avg(array((1, 2, 3))) == 2)  # unweighted average of array
    assert(5 < utils.avg([utils.if_dropout(arange(0, 10), 0.7, True).nonzero()[0].shape[0] for _ in range(100)]) < 9)  # stochastic function therefore needs a range of values

    # activation_funcs.py
    tanh = activation_funcs.Tanh
    leaky = activation_funcs.LeakyReLU
    assert(tanh.predict(1) == 0.7615941559557649)  # formulae
    assert(tanh.gradient(1) == 0.41997434161402614)
    assert(leaky.predict(1, 0.1) == 1)
    assert(leaky.gradient(1, 0.1) == 1)

    # decision_trees.py
    dt = decision_trees.DecisionTree()
    temp_data = array([1, 1, 1, 2, 3, 4])
    assert(dt.entropy(temp_data) == 1.7924812445897977)  # formula
    assert((dt.get_mask(temp_data, 1)[0] == [0, 1, 2]).all())  # finds all indices where element == 1
    assert((dt.get_prob(temp_data) == [1/2, 1/6, 1/6, 1/6]).all())  # frequency of elements

    print("No errors")

