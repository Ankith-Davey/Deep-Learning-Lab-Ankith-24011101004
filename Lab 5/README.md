# Experiment 5 — Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

CS3807 Deep Learning Laboratory, AY 2026-27

## Dataset

Oxford-IIIT Pet Dataset — 37 cat and dog breeds across 7,349 images. Downloaded directly via torchvision datasets, resized to 224×224×3 RGB. Split into 2,944 training images, 736 validation images, and 3,669 test images. Suitable for transfer learning studies.

## Dependencies

torch, torchvision, tensorflow, keras, numpy, matplotlib, scikit-learn (for evaluation metrics — precision, recall, F1, confusion matrix)

## How to run

Open `Experiment_5_Transfer_Learning_and_Hyperparameter_Tuning.ipynb` in Google Colab on a GPU runtime (Runtime → Change runtime type → T4 GPU). Run cells sequentially. The dataset download, five-fold cross-validation across all configurations, and full retraining on the test set take 15–20 minutes on GPU; significantly longer expected on CPU.

## Contents

* `Experiment_5_Transfer_Learning_and_Hyperparameter_Tuning.ipynb`: dataset loading, weight initialization comparison (zero, random, Xavier, He), regularization study (L2, Dropout, Batch Normalization), optimizer comparison (SGD, Momentum, RMSProp, Adam), hyperparameter sweeps (learning rate, batch size, dropout), transfer learning (feature extraction vs. fine-tuning), 5-fold cross-validation across four configurations, final model evaluation on held-out test set, and optional additional exercise with two new configurations.
* `Experiment_5_Report.tex` / `Experiment_5_Report.pdf`: LaTeX source and compiled report, with full analysis of all 15 plots plus discussion answers.
* `Experiment_5_Plots/`: all 15 plots generated — initialization loss/accuracy, regularization accuracy/loss, batch norm comparison, optimizer comparison, learning rate sweep, batch size sweep, dropout sweep, transfer learning comparison, transfer learning loss, cross-validation accuracy, confusion matrix, and misclassified examples — exported as `.png`.

## Key Results

* **Final Model (C2):** He initialization, dropout 0.25, Batch Normalization, Adam optimizer, lr = 0.001
* **Cross-Validation Accuracy:** 91.17% ± 1.26% (5-fold mean ± SD)
* **Test Accuracy:** 89.83%
* **F1-score:** 0.8973
* **Parameters:** 338,469 (frozen MobileNetV2 backbone + new classification head)
* **Training Time:** 12.4 seconds on GPU
* **Main Finding:** Initialization and optimizer choice were far more impactful than regularization alone; fine-tuning underperformed frozen-head feature extraction on a 5-epoch cross-validation budget due to insufficient training time for backbone convergence.
