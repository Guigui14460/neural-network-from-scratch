import numpy as np

from .layer import Layer


class Input(Layer):
    def forward(self, inputs: np.array, training: bool) -> None:
        self.output = inputs

    def backward(self, dvalues: np.array) -> None:
        pass
