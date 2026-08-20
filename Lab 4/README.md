# Experiment 4 — Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

CS3807 Deep Learning Laboratory, AY 2026-27

## Dataset

CIFAR-10 — 50,000 training images, 10,000 test images, 32×32×3 RGB, 10
classes. Downloaded directly via the Kaggle API (`pankrzysiu/cifar10-python`
dataset), which distributes CIFAR-10 in its original pickled batch format
(`data_batch_1`–`data_batch_5`, `test_batch`, `batches.meta`) — no manual
upload needed. Requires a `kaggle.json` API token uploaded to the Colab
sidebar first (Kaggle → Settings → API → Create New Token).

## Dependencies

numpy, pandas, tensorflow, kaggle, matplotlib, scikit-learn (for evaluation
metrics — precision, recall, F1, confusion matrix)

## How to run

Open `Experiment_4_epochs10_20.ipynb` in Google Colab on a GPU runtime
(Runtime → Change runtime type → T4 GPU). Upload `kaggle.json` to the Colab
sidebar when prompted, then run cells sequentially. Total runtime is
roughly 80–130 minutes on GPU — the primary model's transfer learning and
fine-tuning (~20 min), the from-scratch LeNet-5/AlexNet runs (~10 min
combined), the VGG16/ResNet50/InceptionV3 transfer-learning runs (~35 min
combined), and the hyperparameter study (~35 min for 9 short runs) make up
most of that. A much longer run should be expected on CPU. All generated
plots are zipped and downloaded automatically at the end of the notebook so
nothing is lost if the runtime disconnects.

## Contents

* `Experiment_4_epochs10_20.ipynb`: dataset loading (Kaggle pickle format),
  MobileNetV2 transfer learning + fine-tuning pipeline (primary model, full
  Task 2–5 workflow), VGG16/ResNet50/InceptionV3 frozen-base transfer
  learning, LeNet-5/AlexNet trained from scratch, the one-factor-at-a-time
  hyperparameter study (learning rate, batch size, epochs, optimizer, dense
  units, frozen layers), evaluation (accuracy/precision/recall/F1/confusion
  matrix), and the final architecture comparison table
* `Experiment_4_Report.tex` / `Experiment_4_Report.pdf`: LaTeX source and
  compiled report, with full analysis and discussion
* `plots/`: all 9 mandatory plots (sample images, training/validation
  accuracy, training/validation loss, before/after fine-tuning comparison,
  confusion matrix, misclassified images, hyperparameter study) exported as
  `.png` at report quality (Times New Roman, bold axes)
* `hyperparameter_study_results.csv`: raw validation accuracy and training
  time for each of the 9 hyperparameter variants
* `section_18_2_comparison_table.csv`: final accuracy/parameter
  count/training time for each architecture (LeNet-5, AlexNet, VGG16,
  GoogleNet$^{*}$, ResNet50, MobileNetV2)

## Notes on scope

* **GoogleNet** has no officially available pretrained weights in any
  mainstream library, so its row in the comparison table uses
  **InceptionV3** (Google's real successor architecture) as a labeled
  stand-in via the same frozen-base transfer learning procedure — reported
  explicitly as such, not as literal GoogleNet.
* **VGG16 and ResNet50** were run through frozen-base transfer learning only
  (no fine-tuning stage); the full transfer-learning-plus-fine-tuning
  pipeline (Tasks 2–5) was applied only to the primary model, MobileNetV2.
* The **hyperparameter study** varies one parameter at a time off a fixed
  baseline rather than running the full 96-combination grid implied by the
  manual's table, since the full grid would take an estimated 15–20+ hours
  — not practical within a single Colab session.
