import numpy as np

from neural_networks.layers import Layer
from .optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 0.001, decay: float = 0.,
                 epsilon: float = 1e-7, rho: float = .9) -> None:
        Optimizer.__init__(self, learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
        self.rho = rho

    def update_params(self, layer: Layer) -> None:
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * \
            layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
