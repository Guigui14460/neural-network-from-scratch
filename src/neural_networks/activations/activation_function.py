import abc
from typing import Union

import numpy as np


class ActivationFunction(abc.ABC):
    """Class which describe an activation function for a layer of neurons.

    Attributes:
    -----------
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
                tensor of values used to derivate the inputs
        """
        pass

    @abc.abstractmethod
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
        pass
