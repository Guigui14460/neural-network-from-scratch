import numpy as np

from .loss_function import LossFunction


class MeanAbsoluteError(LossFunction):
    def forward(self, output: np.array, y: np.array) -> None:
        return np.mean(np.abs(y - output), axis=-1)

    def backward(self, dvalues: np.array, y: np.array) -> None:
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y - dvalues) / outputs
        self.dinputs /= samples
