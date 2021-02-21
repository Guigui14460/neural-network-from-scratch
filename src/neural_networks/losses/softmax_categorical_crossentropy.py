import numpy as np

from neural_networks.activations import Softmax
from neural_networks.losses.categorical_cross_entropy import LossCategoricalCrossentropy


class SoftmaxCategoricalCrossentropy:
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples
