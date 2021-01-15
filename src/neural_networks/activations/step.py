import numpy as np

from .activation_function import ActivationFunction


class Step(ActivationFunction):
    """Class which describe an activation function for a layer of neurons.

    Attributes:
    -----------
        output: np.array
            tensor representing the result of the layer of neurons
    """

    def forward(self, inputs: np.array, training: bool) -> None:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            inputs: np.array
                tensor of data or coming from a previous layer
        """
        self.inputs = inputs
        self.output = np.float32(inputs > 0)

    def backward(self, dvalues: np.array) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs > 0] = 1
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs: np.array) -> None:
        return outputs
