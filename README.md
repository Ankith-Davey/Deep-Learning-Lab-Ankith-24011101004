# Deep Learning Laboratory

Lab experiments for CS3807 -- Deep Learning Laboratory, B.Tech Artificial Intelligence & Data Science.

## Structure

```
├── Lab 1/   # Single Layer Perceptron
├── Lab 2/   # Multi-Layer Perceptron
├── Lab 3/   # Convolutional Neural Network
├── Lab 4/   # Transfer Learning (CNN Architecture Comparison)
├── Lab 5/   # CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation
├── Lab 6/   # RNN, LSTM and GRU for Sequence Learning and Video Understanding
├── Lab 7/   # Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders
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

## Lab 4 -- Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

- Comparative study of deep CNN architectures on CIFAR-10 using TensorFlow/Keras: LeNet-5 and AlexNet trained from scratch, VGG16/ResNet50/InceptionV3 via frozen-base transfer learning, and MobileNetV2 taken through a full transfer-learning-plus-fine-tuning pipeline.
- One-factor-at-a-time hyperparameter study (learning rate, batch size, epochs, optimizer, dense units, frozen layers) against a fixed baseline.
- GoogleNet's row in the architecture comparison uses InceptionV3 as a labeled stand-in, since no pretrained GoogleNet weights exist in any mainstream library.

## Lab 5 -- Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

- Systematic study of weight initialization strategies (zero, random, Xavier, He), regularization techniques (L2, Dropout, Batch Normalization), and optimization algorithms (SGD, Momentum, RMSProp, Adam) on the Oxford-IIIT Pet dataset (37 cat and dog breeds) using MobileNetV2.
- Hyperparameter sweeps on learning rate, batch size, and dropout rate, with one-factor-at-a-time analysis to isolate effects.
- Transfer learning study comparing frozen-base feature extraction vs. fine-tuning with partial backbone unfreezing, demonstrating that insufficient training epochs limit fine-tuning convergence under cross-validation constraints.
- Model selection via 5-fold cross-validation across four configurations, with final evaluation on a held-out test set. Final model achieved 91.17\% $\pm$ 1.26\% cross-validation accuracy and 89.83\% test accuracy, demonstrating that initialization and optimizer choice had substantially larger impact than regularization techniques on frozen-head transfer learning.

## Lab 6 -- End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

- Sequence classification on the UCI Human Activity Recognition (HAR) dataset using raw inertial signals (128 timesteps x 9 channels), comparing Vanilla RNN, LSTM and GRU under identical preprocessing, splits, and training configuration.
- Backpropagation Through Time (BPTT) covered via a manual numerical exercise alongside the trained models, motivating the vanishing/exploding gradient comparison between the plain RNN and the gated architectures.
- Additional exercises varying recurrent unit count (16/32/64), stacking a second recurrent layer, and comparing bidirectional vs. unidirectional LSTM, plus a sequence-length ablation (T = 32, 64, 128).
- Video understanding extension: a 5-class UCF101 subset (via Kaggle mirror) processed through a frozen, pretrained MobileNetV2 feature extractor followed by an LSTM/GRU classifier, with softmax confidence scores reported per prediction.
- A synthetic sequence-to-sequence reversal task (encoder--decoder LSTM with teacher forcing) demonstrating token vs. sequence accuracy. LSTM and GRU reached 93.06\% and 92.78\% test accuracy respectively on HAR (vs. 75.00\% for the plain RNN), while CNN--GRU reached 100\% and CNN--LSTM 75.00\% on the video task, showing that the LSTM/GRU ordering was not fixed across configurations.

## Lab 7 -- End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders

- Image reconstruction, denoising and generative modelling on MNIST (10,000 training and 2,000 test images, fixed seed 109) using TensorFlow/Keras, with the same data subsets reused across all models for a controlled comparison.
- Fully connected autoencoder (16-dimensional latent) compared against a convolutional autoencoder under MSE, MAE and SSIM, plus a UpSampling2D vs. Conv2DTranspose decoder comparison and latent-capacity sweeps for both the fully connected (2/8/16/32) and convolutional (4/8/16/32 channels) models.
- Denoising convolutional autoencoder trained against Gaussian (sigma = 0.1/0.2/0.3) and salt-and-pepper (p = 0.05/0.10/0.20) corruption with the clean image as target.
- Variational autoencoder with reparameterization trick and 2D latent space visualization, extended with latent dimension 8, a beta-VAE ablation (beta = 0.5/1.0/5.0), random generation (25 and 100 samples), and latent interpolation including centroid-based interpolation between the most separated digit classes.
- Per-image reconstruction error distribution and high-error sample analysis for the fully connected autoencoder. The convolutional autoencoder reached 0.9751 SSIM against 0.7452 for the fully connected autoencoder using roughly a third of the parameters, while the VAE (latent 2) had the weakest reconstruction (SSIM 0.5007) in exchange for a structured, samplable latent space in which only digits 0 and 1 separated cleanly.

## Author

Ankith U Davey -- 24011101004
