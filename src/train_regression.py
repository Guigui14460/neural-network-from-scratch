import nnfs
import nnfs.datasets

from neural_networks import Model
from neural_networks.accuracies import RegressionAccuracy
from neural_networks.activations import ReLU, Linear
from neural_networks.layers import Dense
from neural_networks.losses import MeanSquaredError
from neural_networks.optimizers import Adam

nnfs.init()


X, y = nnfs.datasets.sine_data()
X_test, y_test = nnfs.datasets.sine_data()

model = Model()

model.add(Dense(1, 64))
model.add(ReLU())
model.add(Dense(64, 64))
model.add(ReLU())
model.add(Dense(64, 1))
model.add(Linear())
model.set(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=.005, decay=1e-3),
    accuracy=RegressionAccuracy()
)

model.finalize()
model.train(X, y, epochs=10000, print_every=100,
            validation_data=(X_test, y_test))
