import cv2
import nnfs
import numpy as np

from neural_networks import Model

nnfs.init()

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


model = Model.load("fashion_mnist.model")

# test with custom clothes
image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)

image_data2 = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)
image_data2 = cv2.resize(image_data2, (28, 28))
image_data2 = 255 - image_data2
image_data2 = (image_data2.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

confidences = model.predict(image_data2)
predictions = model.output_layer_activation.predictions(confidences)
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)
