import numpy as np

from .activation_function import ActivationFunction


class Softmax(ActivationFunction):
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
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues: np.array) -> None:
        """Make the gradient with given derivative value.

        Parameters:
        -----------
            dvalues: np.array
                tensor of values used to derivate the inputs
        """
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs: np.array) -> None:
        return np.argmax(outputs, axis=1)
