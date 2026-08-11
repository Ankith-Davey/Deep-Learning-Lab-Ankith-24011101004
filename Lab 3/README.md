# Experiment 3 — CNN for Image Classification

CS3807 Deep Learning Laboratory, AY 2026-27

## Dataset

CIFAR-10 — 50,000 training images, 10,000 test images, 32×32×3 RGB, 10
classes. Downloaded directly via the Kaggle API (`kagglehub`,
`pankrzysiu/cifar10-python` dataset) — no manual upload needed. Requires
a `kaggle.json` API token uploaded to the Colab sidebar first (Kaggle →
Settings → API → Legacy API Credentials → Create Legacy API Key).

## Dependencies

numpy, tensorflow, kagglehub, matplotlib, scikit-learn (for evaluation
metrics only — precision, recall, F1, confusion matrix)

## How to run

Open `Experiment_3_CNN_Deep_learning_lab_3.ipynb` in Google Colab on a
GPU runtime (Runtime → Change runtime type → T4 GPU). Upload
`kaggle.json` to the Colab sidebar, then run cells sequentially. The
dataset download and full 20-epoch training run take a few minutes on
GPU; a much longer run should be expected on CPU.

## Contents

* `Experiment_3_CNN_Deep_learning_lab_3.ipynb`: dataset loading, kernel
  size and stride/padding comparisons, feature map visualization (before
  and after training), max vs. average pooling comparison, CNN
  construction and training, evaluation, and the 5 additional exercises
  (output size/parameter calculations, ReLU vs. Sigmoid, filter count
  comparison)
* `CNN_Experiment_3_Report.tex` / `CNN_Experiment_3_Report.pdf`: LaTeX
  source and compiled report, with full analysis
* `Experiment_3_Plots/`: all 11 plots generated — the 8 mandatory plots
  (feature maps split into separate before/after-training files) plus 2
  additional exercise plots (pooling comparison, ReLU vs. Sigmoid) —
  exported as `.png`
