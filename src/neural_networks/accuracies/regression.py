import numpy as np

from .accuracy import Accuracy


class RegressionAccuracy(Accuracy):
    def __init__(self) -> None:
        self.precision = None

    def init(self, y, reinit: bool = False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.absolute(predictions - y) < self.precision
