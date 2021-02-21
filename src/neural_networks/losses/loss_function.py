import abc

import numpy as np


class LossFunction(abc.ABC):
    """Class which describe the base of a loss function."""

    def regularization_loss(self) -> float:
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(np.abs(layer.weights * layer.weights))
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(np.abs(layer.biases * layer.biases))
        return regularization_loss

    def new_pass(self) -> None:
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, output: np.ndarray, y: np.ndarray, *, include_regularization: bool = False) -> tuple:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            output
                tensor of output data
            y
                represents the good values (labels)

        Returns:
        --------
            result
                mean of the loss function of the model
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization: bool = False) -> tuple:
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return (data_loss,)
        return data_loss, self.regularization_loss()

    def remember_trainable_layers(self, trainable_layers: list) -> None:
        self.trainable_layers = trainable_layers

    @abc.abstractmethod
    def forward(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            output
                tensor of output data
            y
                represents the good values (labels)

        Returns:
        --------
            result
                tensor representing the loss for each output neurons of the layer
        """
        pass

    @abc.abstractmethod
    def backward(self, dvalues: np.ndarray, y: np.ndarray) -> None:
        """Make the gradient with given derivative value.

        Parameters:
        -----------
            dvalues
                tensor of values used to derivate the output
            y
                represents the good values (labels)
        """
        pass
