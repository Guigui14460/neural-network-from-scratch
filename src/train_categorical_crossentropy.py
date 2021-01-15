import nnfs
import nnfs.datasets

from neural_networks import Model
from neural_networks.accuracies import CategoricalCrossentropyAccuracy
from neural_networks.activations import ReLU, Softmax
from neural_networks.layers import Dense, Dropout
from neural_networks.losses import LossCategoricalCrossentropy
from neural_networks.optimizers import Adam

nnfs.init()


X, y = nnfs.datasets.spiral_data(samples=1000, classes=3)
X_test, y_test = nnfs.datasets.spiral_data(samples=100, classes=3)

model = Model()

model.add(Dense(2, 512, weight_regularizer_l2=5e-4,
                bias_regularizer_l2=5e-4))
model.add(ReLU())
model.add(Dropout(0.1))
model.add(Dense(512, 3))
model.add(Softmax())
model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=Adam(learning_rate=.05, decay=5e-5),
    accuracy=CategoricalCrossentropyAccuracy()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10000, print_every=100)
