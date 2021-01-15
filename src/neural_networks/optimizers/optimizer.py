import abc

from neural_networks.layers import Layer


class Optimizer(abc.ABC):
    def __init__(self, learning_rate: float = 1., decay: float = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def post_update_params(self) -> None:
        self.iterations += 1

    @abc.abstractmethod
    def update_params(self, layer: Layer) -> None:
        pass
