# Neural Network From Scratch

The entire of the code has been taken from the [https://nnfs.io/](NNFS) book written by Sentdex.

## Table of contents

2. [Setup](#setup)
1. [Usage of the application](#usage)

## Setup

To run the project, you need to use `pipenv`.
You can install all dependencies with this command :
```shell
$ pipenv install
```

## Usage

You can use differents files to test the neural network.

To train simple regression :
```shell
$ python src/train_regression.py
```

To train binary logistic regrsssion :
```shell
$ python src/train_binary_logistic_regression.py
```

To train categorical crossentropy :
```shell
$ python src/train_categorical_crossentropy.py
```

For categorical exemple :
```shell
$ python src/real_categorical_example.py
$ python src/load_real_categorical_example.py
```
