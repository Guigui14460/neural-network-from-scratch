import numpy as np

from .layer import Layer


class Input(Layer):
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.output = inputs

    def backward(self, dvalues: np.ndarray) -> None:
        pass
