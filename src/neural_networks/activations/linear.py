import numpy as np

from .activation_function import ActivationFunction


class Linear(ActivationFunction):
    """Class which describe an activation function for a layer of neurons.

    Attributes:
    -----------
        output
            tensor representing the result of the layer of neurons
    """

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            inputs
                tensor of data or coming from a previous layer
        """
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues.copy()

    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return outputs
