# Deep Learning Laboratory

Lab experiments for CS3807 -- Deep Learning Laboratory, B.Tech Artificial Intelligence & Data Science.

## Structure

```
├── Lab 1/   # Single Layer Perceptron
├── Lab 2/   # Multi-Layer Perceptron
├── Lab 3/   # Convolutional Neural Network
└── README.md
```

## Lab 1 -- Single Layer Perceptron (SLP)

- Binary classification on the UCI Banknote Authentication dataset.
- Perceptron learning algorithm implemented from scratch (NumPy).
- Includes SLP training on the AND, OR, and NOT logic gates to verify the implementation on linearly separable data.

## Lab 2 -- Multi-Layer Perceptron (MLP)

- Multi-class image classification on Fashion-MNIST using TensorFlow/Keras.
- Hyperparameter optimization via RandomizedSearchCV + SciKeras.
- Includes an MLP implemented from scratch (NumPy) to solve the XOR problem, demonstrating why a single-layer perceptron fails on non-linearly separable data.

## Lab 3 -- Convolutional Neural Network (CNN)

- Multi-class image classification on CIFAR-10 using TensorFlow/Keras.
- Convolution, pooling, and feature map visualization studied directly: kernel size, stride/padding, and max vs. average pooling comparisons.
- Includes hyperparameter-style comparisons on activation function (ReLU vs. Sigmoid) and filter count (16 vs. 64).

## Author

Ankith U Davey -- 24011101004
