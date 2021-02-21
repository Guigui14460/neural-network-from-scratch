import numpy as np

from .accuracy import Accuracy


class CategoricalCrossentropyAccuracy(Accuracy):
    def init(self, y: np.ndarray, reinit: bool = False):
        pass

    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
