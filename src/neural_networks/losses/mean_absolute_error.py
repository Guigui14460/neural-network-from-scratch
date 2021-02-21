import numpy as np

from .loss_function import LossFunction


class MeanAbsoluteError(LossFunction):
    def forward(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(y - output), axis=-1)

    def backward(self, dvalues: np.ndarray, y: np.ndarray) -> None:
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.true_divide(np.sign(y - dvalues), outputs)
        self.dinputs = np.true_divide(self.dinputs, samples)
