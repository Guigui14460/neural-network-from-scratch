import numpy as np

from neural_networks.layers import Layer
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate: float = .001, decay: float = 0., epsilon: float = 1e-7,
                 beta_1: float = .9, beta_2: float = 0.999) -> None:
        Optimizer.__init__(self, learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer: Layer) -> None:
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momemtums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momemtums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momemtums = self.beta_1 * \
            layer.weight_momemtums + (1 - self.beta_1) * layer.dweights
        layer.bias_momemtums = self.beta_1 * \
            layer.bias_momemtums + (1 - self.beta_1) * layer.dbiases

        weight_momemtums_corrected = layer.weight_momemtums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momemtums_corrected = layer.bias_momemtums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momemtums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momemtums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
