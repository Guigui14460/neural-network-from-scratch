import abc

import numpy as np


class Layer(abc.ABC):
    """Class which describe a layer of multiple neurons.

    Attributes:
    -----------
        weights
            tensor describe the weight of each value of input data
        biases
            tensor describe some irrigularities for each value of input data
        output
            tensor representing the result of the layer of neurons
    """
    @abc.abstractmethod
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            inputs
                tensor of data or coming from a previous layer
        """
        pass

    @abc.abstractmethod
    def backward(self, dvalues: np.ndarray) -> None:
        """Make the gradient with given derivative value.

        Parameters:
        -----------
            dvalues
                tensor of values used to derivate the weights, biases and inputs
        """
        pass
