import numpy as np

from .accuracy import Accuracy


class RegressionAccuracy(Accuracy):
    def __init__(self) -> None:
        self.precision = None

    def init(self, y: np.array, reinit: bool = False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions: np.array, y: np.array) -> np.array:
        return np.absolute(predictions - y) < self.precision
