import abc

import numpy as np


class Accuracy(abc.ABC):
    def new_pass(self) -> None:
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, predictions: np.ndarray, y: np.ndarray) -> float:
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self) -> float:
        return self.accumulated_sum / self.accumulated_count

    @abc.abstractmethod
    def init(self, y: np.ndarray, reinit: bool = False):
        pass

    @abc.abstractmethod
    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass
