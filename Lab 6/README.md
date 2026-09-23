# Experiment 6 — End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

CS3807 Deep Learning Laboratory, AY 2026-27

## Dataset

UCI Human Activity Recognition Using Smartphones — raw inertial signal files (9 channels: 3-axis body acceleration, 3-axis gyroscope, 3-axis total acceleration), each window 128 timesteps. Downloaded directly from the UCI ML Repository. Train and test partitions combined into a single pool, then a balanced subset of 400 windows per class (2,400 windows total, 6 classes) sampled and split 70/15/15 into training (1,680), validation (360) and test (360) sets using a fixed random seed (109). For the video understanding extension, a pre-curated 5-class subset of UCF101 (CricketShot, PlayingCello, Punch, ShavingBeard, TennisSwing) via a Kaggle mirror, 10 videos sampled per class (50 videos total, 10 frames per video resized to 224×224×3).

## Dependencies

tensorflow, keras, numpy, pandas, matplotlib, scikit-learn (for evaluation metrics — precision, recall, F1, confusion matrix), opencv-python-headless (for video frame extraction)

## How to run

Open `Deep_Learning_Lab_6_Code.ipynb` in Kaggle on a GPU runtime (T4 GPU). Run cells sequentially. The video understanding section requires attaching the `abdallahwagih/ucf101-videos` Kaggle dataset via "Add Input" before running that block. Dataset download, RNN/LSTM/GRU training, the sequence-length and additional-exercise sweeps, video feature extraction and training, and the seq2seq task together take roughly 30–40 minutes on GPU; significantly longer expected on CPU.

## Contents

* `Deep_Learning_Lab_6_Code.ipynb`: preprocessing and temporal visualization, BPTT numerical exercise, Vanilla RNN / LSTM / GRU implementation and training, performance evaluation and confusion matrices, sequence-length ablation (T = 32, 64, 128), additional exercises (unit-count variants, GRU vs. LSTM at matched units, second recurrent layer, bidirectional LSTM), video understanding pipeline (CNN feature extraction with MobileNetV2 + LSTM/GRU classification), and a synthetic sequence-to-sequence reversal task with an encoder–decoder LSTM.
* `Experiment_6_Report.tex` / `Experiment_6_Report.pdf`: LaTeX source and compiled report, with full analysis of all 9 plots plus discussion answers.
* `Experiment 6 Plots/`: all plots generated — sensor signal visualization, training/validation loss and accuracy curves for RNN/LSTM/GRU, confusion matrices, model performance comparison, sequence-length vs. F1, unit-count vs. F1, video sample frames, video training curves and video confusion matrices — exported as `.png`.

## Key Results

* **RNN:** 75.00% test accuracy, macro F1 = 74.97%, 1,974 parameters, 24.2s training time
* **LSTM:** 93.06% test accuracy, macro F1 = 93.02%, 6,006 parameters, 20.8s training time
* **GRU:** 92.78% test accuracy, macro F1 = 92.78%, 4,758 parameters, 18.0s training time
* **Video CNN–LSTM:** 75.00% test accuracy, F1 = 76.00%, 168,229 parameters
* **Video CNN–GRU:** 100.00% test accuracy and F1, 126,309 parameters
* **Seq2Seq Reversal:** 100.00% token accuracy, 100.00% sequence accuracy, final validation loss 0.0063
* **Main Finding:** LSTM and GRU both handled the 128-step HAR sequences comfortably and closely matched each other, while the plain RNN was the clear underperformer with less reliable behaviour across configurations (non-monotonic F1 swings across sequence lengths); SITTING and STANDING were the most confused activity pair in every model due to their similar low-variance static sensor signatures; and the LSTM/GRU ordering itself was not fixed, flipping across the different unit-count, depth and directionality configurations tested in the additional exercises.
