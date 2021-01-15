import numpy as np

from .loss_function import LossFunction


class MeanSquaredError(LossFunction):
    def forward(self, output: np.array, y: np.array) -> None:
        return np.mean((y - output) ** 2, axis=-1)

    def backward(self, dvalues: np.array, y: np.array) -> None:
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y - dvalues) / outputs
        self.dinputs /= samples
