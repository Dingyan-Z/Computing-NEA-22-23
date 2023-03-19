from decision_trees import RandomForest
from regression import Linear, Logistic
from reinforcement_learning import RL
from numpy import split, nonzero, array, atleast_1d
from os.path import exists
from os import getcwd
from csv import reader, writer
from pickle import dump, load
from utils import sep
from numpy.random import shuffle
from sys import stdout
from matplotlib.pyplot import ylabel, plot, show, xlabel, title
from neural_net import Dense


def force_inp(prompt, type_, bound=None):
    while True:
        try:
            inp = type_(input(prompt))
            if bound is None or bound[0] <= inp < bound[1]:
                return inp
            print("Value out of bound")
        except ValueError:
            print(f"Please input as {type_.__name__}")


def inp_path(prompt):
    while True:
        path = force_inp(prompt, str)
        if exists(path):
            return path
        print("File not found")


def menu(msg, choices):
    print(msg)
    for i, v in enumerate(choices):
        print(f"{i + 1}. {v}")
    choice = force_inp("Choice: ", int, (1, len(choices) + 1))
    print(f"\n{choices[choice - 1]}")
    return choice


if __name__ == '__main__':
    print("""Instructions
>> Data should be in a csv file without missing values (Header, i.e. first row, will be ignored)
>> Dense Neural Networks, Linear Regression handle non-string values
>> Random Forest handle discrete values
>> Logistic regression handles non-string values with binary, i.e. True/False, output
""")
    mode = menu("Mode", ("Load", "Train"))
    with open(inp_path("Data path: ")) as file:
        data = array(list(reader(file))[1:])
    split_data = split(data, [int(len(data) * 0.8)])
    model = action = algorithm = None
    try:
        match mode:
            case 1:
                with open(inp_path("Model path: "), "rb") as file:
                    model = load(file)
                action = menu("Action", ("Predict", "Retrain")) if type(model) == Dense else 1
                if action == 1:
                    print("\nPredict")
                    predictions = model.predict(data if type(model) == RandomForest else data.astype(float))
                    print("Done")
                    count = 0
                    while True:
                        save_path = f"{getcwd()}/predictions_output/{type(model).__name__}{count}.csv"
                        if not exists(save_path):
                            with open(save_path, "w", newline="") as file:
                                writer(file, delimiter=" ").writerows([atleast_1d(v) for v in predictions])
                            print("\nOutput")
                            print(f"File saved to {save_path}")
                            break
                        count += 1
            case 2:
                algorithm = (RandomForest, Linear, Logistic, RL)[menu("\nAlgorithms", ("Random Forest", "Linear Regression", "Logistic Regression", "Neural Network")) - 1]
                shuffle(data)
                split_data = split(data, [int(len(data) * 0.8)])
                match algorithm.__name__:
                    case RandomForest.__name__:
                        model = RandomForest()
                        model.train(split_data[0])
                        feats, labels = sep(split_data[1])
                        results = model.predict(feats)
                        print("\nAccuracy")
                        print(nonzero(results == labels.T[0])[0].shape[0] / results.shape[0] * 100, "%")
                    case Linear.__name__ | Logistic.__name__:
                        model = algorithm(split_data[0].shape[1] - 1)
                        print("\nProgress")
                    case RL.__name__:
                        rl_epochs = 1
                        rl = RL(*[v.astype(float) for v in split_data])
                        print("\nProgress")
                        for epoch in range(rl_epochs):
                            stdout.write("\b" * rl_epochs)
                            stdout.write(f"{epoch}/{rl_epochs}")
                            rl.train()
                        model = rl.get_net()
                        print()
        if (type(model) in [Linear, Logistic, Dense] and action != 1) or action == 2:
            epochs = 1
            training_data, test_data = [v.astype(float) for v in split_data]
            title(f"{type(model).__name__} Cost")
            xlabel("Epochs")
            ylabel("Cost")
            cost_history = []
            for epoch in range(epochs):
                if epoch % 200 == 0:
                    stdout.write("\b" * epochs)
                    stdout.write(f"{epoch}/{epochs}")
                model.train(*sep(training_data))
                cost_history.append(model.cost(*sep(test_data)))
            plot(range(len(cost_history)), cost_history)
            show()
            print()
        count = 0
        if mode == 2:
            while True:
                save_path = f"{getcwd()}/model_output/{algorithm.__name__}{count}.txt"
                if not exists(save_path):
                    with open(save_path, "wb") as file:
                        dump(model, file)
                    print("\nOutput")
                    print(f"File saved to {save_path}")
                    print(f"All models use pseudo-randomness therefore more than 1 training attempts may be needed")
                    if algorithm == RL:
                        print(f"To retrain without optimising hyper-parameters, load the model")
                    break
                count += 1
    except Exception as e:
        print(e)
        print("\n\nInvalid data file format")
