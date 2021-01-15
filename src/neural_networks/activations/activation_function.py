import abc

import numpy as np


class ActivationFunction(abc.ABC):
    """Class which describe an activation function for a layer of neurons.

    Attributes:
    -----------
        output: np.array
            tensor representing the result of the layer of neurons
    """
    @abc.abstractmethod
    def forward(self, inputs: np.array, training: bool) -> None:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            inputs: np.array
                tensor of data or coming from a previous layer
        """
        pass

    @abc.abstractmethod
    def backward(self, dvalues: np.array) -> None:
        """Make the gradient with given derivative value.

        Parameters:
        -----------
            dvalues: np.array
                tensor of values used to derivate the inputs
        """
        pass

    @abc.abstractmethod
    def predictions(self, outputs: np.array) -> None:
        pass
