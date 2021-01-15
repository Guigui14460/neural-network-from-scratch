import numpy as np

from .layer import Layer


class Dense(Layer):
    """Describe a fully connected hidden neurons from data to output.

    Attributes:
    -----------
        weights: np.array
            tensor describe the weight of each value of input data
        biases: np.array
            tensor describe some irrigularities for each value of input data
        output: np.array
            tensor representing the result of the layer of neurons
    """

    def __init__(self, n_inputs: int, n_neurons: int, weight_regularizer_l1: float = 0,
                 weight_regularizer_l2: float = 0, bias_regularizer_l1: float = 0,
                 bias_regularizer_l2: float = 0) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs: np.array, training: bool) -> None:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            inputs: np.array
                tensor of data or coming from a previous layer
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.array) -> None:
        """Make the gradient with given derivative value.

        Parameters:
        -----------
            dvalues: np.array
                tensor of values used to derivate the weights, biases and inputs
        """
        # partial derivation of weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradients on regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self) -> tuple:
        return self.weights, self.biases

    def set_parameters(self, weights: np.array, biases: np.array) -> None:
        self.weights = weights
        self.biases = biases
