from typing import Union

import numpy as np

from .activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    """Class which describe an activation function for a layer of neurons.
    The activation function is the sigmoid function.

    Attributes:
    -----------
        output: ndarray
            tensor representing the result of the layer of neurons
    """

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            inputs: ndarray
                tensor of data or coming from a previous layer
        """
        self.inputs = inputs
        self.output = np.true_divide(1, np.add(1, np.exp(-inputs)))

    def backward(self, dvalues: np.ndarray) -> None:
        """Make the gradient with given derivative value.

        Parameters:
        -----------
            dvalues: ndarray
                tensor of values used to derivate the inputs
        """
        self.dinputs = dvalues * np.subtract(1, self.output) * self.output

    def predictions(self, outputs: np.ndarray) -> Union[np.ndarray, float, int]:
        """Make predictions with given layer output value.

        Parameters:
        -----------
            outputs: ndarray
                tensor of values used to generate the predictions

        Returns:
        --------
            outputs2: Union[ndarray, float, int]
                tensor of values or single value represents the layer predictions
        """
        return (outputs > .5) * 1
