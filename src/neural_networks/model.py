import copy
import pickle

import numpy as np

from neural_networks.accuracies import Accuracy
from neural_networks.activations import Softmax
from neural_networks.layers import Layer, Input
from neural_networks.losses import LossFunction, LossCategoricalCrossentropy, SoftmaxCategoricalCrossentropy
from neural_networks.optimizers import Optimizer


class Model:
    def __init__(self) -> None:
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer) -> None:
        self.layers.append(layer)

    def set(self, *, loss: LossFunction, optimizer: Optimizer, accuracy: Accuracy) -> None:
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self) -> None:
        self.input_layer = Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, LossCategoricalCrossentropy):
            self.softmax_classifier_output = SoftmaxCategoricalCrossentropy()

    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 1, print_every: int = 1,
              validation_data: tuple = None, batch_size: int = None) -> None:
        self.accuracy.init(y)
        train_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}")
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True)
                data_loss, regularization_loss = self.loss.calculate(
                    output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f"Step: {step}, Accuracy: {accuracy:.3f}, Loss: {loss:.3f} (data: {data_loss:.3f}, reg: {regularization_loss:.3f}), Learning rate: {self.optimizer.current_learning_rate}")

        epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(
            include_regularization=True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()
        print(f"Accuracy: {epoch_accuracy:.3f}, Loss: {epoch_loss:.3f} (data: {epoch_data_loss:.3f}, reg: {epoch_regularization_loss:.3f}), Learning rate: {self.optimizer.current_learning_rate}")

        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)

    def forward(self, X: np.ndarray, training: bool) -> np.ndarray:
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output: np.ndarray, y: np.ndarray) -> None:
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray, *, batch_size: int = None) -> None:
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(
            f"Validation: \tAcc: {validation_accuracy:.3f}, Loss: {validation_loss:.3f}")

    def predict(self, X: np.ndarray, *, batch_size: int = None) -> np.ndarray:
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
        return np.vstack(output)

    def get_parameters(self) -> list:
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters: list) -> None:
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path: str) -> None:
        model = copy.deepcopy(self)

        # reset accumulated values
        model.loss.new_pass()
        model.accuracy.new_pass()

        # remove data
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str) -> 'Model':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
