# Experiment 2 — MLP for Multi-Class Image Classification

CS3807 Deep Learning Laboratory, AY 2026-27

## Dataset

Fashion-MNIST — 60,000
training images, 10,000 test images, 28×28 grayscale, 10 classes. Loaded
directly via `keras.datasets.fashion_mnist.load_data()` (no separate
dataset file needed).

## Dependencies

numpy, tensorflow, scikeras, `scikit-learn<1.6` (pinned — scikeras isn't
yet compatible with sklearn's newer tagging API), matplotlib

## How to run

Open the `.ipynb` in Google Colab, run cells sequentially. If Cell 1 is
re-run after `sklearn` has already been imported in the session, restart
the runtime first so the version pin actually takes effect.

## Contents

* Notebook: EDA, preprocessing, MLP construction, baseline training,
  evaluation, automated hyperparameter optimization (RandomizedSearchCV +
  SciKeras, 5-fold CV), optimized model retraining and comparison
* Report: LaTeX-generated PDF with full analysis
* `plots/`: all 9 required plots, exported as `.eps`
