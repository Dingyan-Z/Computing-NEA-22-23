import regression
import numpy as np
import matplotlib.pyplot as plt


class Settings:
    EPOCHS = 100
    MODEL = regression.Logistic
    DATA_RANGE = range(-10, 10)


data = np.array([[i, i] for i in Settings.DATA_RANGE])
# data = np.array([[i, 1 / (1 + np.exp(-i))] for i in Settings.DATA_RANGE])

_, axis = plt.subplots(2)
axis[0].set_title("Predictions")
axis[1].set_title("Cost")

threshold = round(len(data) * 0.8)
training, test = data[:threshold], data[threshold:]
training_data, training_labels = training[:, :-1], training[:, -1]
test_data, test_labels = training[:, :-1], training[:, -1]

model = Settings.MODEL(training_data.shape[1])
costs = []
axis[0].plot(Settings.DATA_RANGE, model.predict(np.array(Settings.DATA_RANGE)))
for _ in range(Settings.EPOCHS):
    model.train(training_data, training_labels)
    costs.append(model.cost(test_data, test_labels))

axis[0].plot(Settings.DATA_RANGE, model.predict(np.array(Settings.DATA_RANGE)))
axis[1].plot(range(Settings.EPOCHS), costs)
axis[0].plot(data[:, 0], data[:, 1], "ro")
axis[0].legend(["Before", "After", "Data"])
plt.show()
