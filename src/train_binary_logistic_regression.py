import nnfs
import nnfs.datasets

from neural_networks import Model
from neural_networks.accuracies import CategoricalCrossentropyAccuracy
from neural_networks.activations import ReLU, Sigmoid
from neural_networks.layers import Dense, Dropout
from neural_networks.losses import LossBinaryCrossentropy
from neural_networks.optimizers import Adam

nnfs.init()


X, y = nnfs.datasets.spiral_data(samples=100, classes=2)
X_test, y_test = nnfs.datasets.spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = Model()
model.add(Dense(2, 64, weight_regularizer_l2=5e-4,
                bias_regularizer_l2=5e-4))
model.add(ReLU())
model.add(Dropout(0.15))
model.add(Dense(64, 1))
model.add(Sigmoid())
model.set(
    loss=LossBinaryCrossentropy(),
    optimizer=Adam(learning_rate=.05, decay=5e-7),
    accuracy=CategoricalCrossentropyAccuracy()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10000, print_every=100)
