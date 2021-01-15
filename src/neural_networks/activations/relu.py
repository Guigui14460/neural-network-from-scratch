import numpy as np

from .activation_function import ActivationFunction


class ReLU(ActivationFunction):
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
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: np.array) -> None:
        """Make the gradient with given derivative value.

        Parameters:
        -----------
            dvalues: np.array
                tensor of values used to derivate the inputs
        """
        self.dinputs = dvalues.copy()
        # zero gradient for negative input values
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs: np.array) -> None:
        return outputs
