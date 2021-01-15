import numpy as np

from neural_networks.layers import Layer
from .optimizer import Optimizer


class StochasticGradientDescent:
    def __init__(self, learning_rate: float = 1., decay: float = 0., momentum: float = 0.) -> None:
        Optimizer.__init__(self, learning_rate=learning_rate, decay=decay)
        self.momentum = momentum

    def update_params(self, layer: Layer) -> None:
        if self.momentum:
            if not hasattr(layer, 'weight_momemtums'):
                layer.weight_momemtums = np.zeros_like(layer.weights)
                layer.bias_momemtums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momemtums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momemtums = weight_updates

            bias_updates = self.momentum * layer.bias_momemtums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momemtums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
