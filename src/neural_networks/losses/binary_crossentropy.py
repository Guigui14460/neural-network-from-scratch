import numpy as np

from .loss_function import LossFunction


class LossBinaryCrossentropy(LossFunction):
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
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        sample_losses = -(y * np.log(output_clipped) +
                          (1-y) * np.log(1-output_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

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
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y/clipped_dvalues - (1-y) /
                         (1-clipped_dvalues)) / outputs
        self.dinputs /= samples
