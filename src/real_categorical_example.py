import nnfs
import numpy as np

from neural_networks import Model
from neural_networks.accuracies import CategoricalCrossentropyAccuracy
from neural_networks.activations import ReLU, Softmax
from neural_networks.layers import Dense
from neural_networks.losses import LossCategoricalCrossentropy
from neural_networks.optimizers import Adam
from data import create_data_mnist

nnfs.init()


print("Loading data ...")
X, y, X_test, y_test = create_data_mnist("fashion_mnist_images")

print("Shuffling data ...")
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

print("Preprocessing data...")
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(
    X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()

model.add(Dense(X.shape[1], 128))
model.add(ReLU())
model.add(Dense(128, 128))
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())
model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=Adam(decay=1e-4),
    accuracy=CategoricalCrossentropyAccuracy()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=10,
            batch_size=128, print_every=100)
model.evaluate(X_test, y_test)
model.save_parameters("fashion_mnist.params")
model.save("fashion_mnist.model")
