import abc

import numpy as np


class Layer(abc.ABC):
    """Class which describe a layer of multiple neurons.

    Attributes:
    -----------
        weights: np.array
            tensor describe the weight of each value of input data
        biases: np.array
            tensor describe some irrigularities for each value of input data
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
                tensor of values used to derivate the weights, biases and inputs
        """
        pass
