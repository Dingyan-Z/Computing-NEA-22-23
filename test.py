import regression
import numpy as np


class Settings:
    EPOCHS = 100
    MODEL = regression.Linear
    DATA_RANGE = range(-20, 21)


data = np.array([[i, i] for i in Settings.DATA_RANGE])
# data = np.array([[i, 1 / (1 + np.exp(-i))] for i in Settings.DATA_RANGE])
np.random.shuffle(data)
threshold = round(len(data) * 0.8)
training, test = data[:threshold], data[threshold:]
training_data, training_labels = training[:, :-1], training[:, -1]
test_data, test_labels = training[:, :-1], training[:, -1]

Settings.MODEL(training_data.shape[1]).train(training_data, training_labels, test_data, test_labels, 100, True)
