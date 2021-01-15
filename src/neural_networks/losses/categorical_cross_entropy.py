import numpy as np

from .loss_function import LossFunction


class LossCategoricalCrossentropy(LossFunction):
    """Class which describe the base of a loss function."""

    def forward(self, output: np.array, y: np.array) -> None:
        """Make the results for all neurons of the layer.

        Parameters:
        -----------
            output: np.array
                tensor of output data
            y: np.array
                represents the good values (labels)

        Returns:
        --------
            result: np.array
                tensor representing the loss for each output neurons of the layer
        """
        samples = len(output)
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        if len(y.shape) == 1:  # categorical labels
            correct_confidences = output_clipped[range(samples), y]
        elif len(y.shape) == 2:  # one-hot encoded labels
            correct_confidences = np.sum(output_clipped * y, axis=1)
        return -np.log(correct_confidences)

    def backward(self, dvalues: np.array, y: np.array) -> None:
        """Make the gradient with given derivative value.

        Parameters:
        -----------
            dvalues: np.array
                tensor of values used to derivate the output
            y: np.array
                represents the good values (labels)
        """
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y.shape) == 1:
            y = np.eye(labels)[y]

        self.dinputs = -y / dvalues
        self.dinputs /= samples
