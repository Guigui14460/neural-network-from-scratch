import numpy as np

from .loss_function import LossFunction


class MeanSquaredError(LossFunction):
    def forward(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.mean(np.square(y - output), axis=-1)

    def backward(self, dvalues: np.ndarray, y: np.ndarray) -> None:
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y - dvalues) / outputs
        self.dinputs /= samples
