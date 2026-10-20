# Experiments 8 & 9 — Effect of Attention Type and Attention Position in a Pretrained ResNet50

CS3807 Deep Learning Laboratory, AY 2026-27

## Dataset

TensorFlow Flowers (`tf_flowers`), 3,670 RGB flower images in five classes (daisy, dandelion, roses, sunflowers, tulips), loaded through `tensorflow_datasets` (with a direct download of the `flower_photos` archive as a fallback). Every image is resized to 224×224×3 and passed through the ResNet50 ImageNet preprocessing (RGB to BGR, mean subtraction). A stratified 70/15/15 split was made once with a fixed random seed (109), giving 2,569 training, 550 validation and 551 test images, and the same samples were reused in all sixteen training runs so all comparisons are controlled; the split indices are saved in `split_indices.json`. Labels are the classification targets. Augmentation is applied to the training images only (horizontal flip, rotation up to about 14 degrees, 10% zoom, 10% translation, 10% contrast change) and is identical for every model. One test image is worth 0.18 accuracy points, which sets the resolution of all comparisons.

## Dependencies

tensorflow, keras, tensorflow-datasets, numpy, pandas, matplotlib, scikit-learn (stratified split, precision/recall/F1 and confusion matrices)

## How to run

Open `Experiment_8&9_code_file.ipynb` in Kaggle on a GPU runtime (T4 GPU) with Internet turned on, because the flowers dataset and the ImageNet ResNet50 weights are downloaded automatically; no dataset needs to be attached. Use Save & Run All (Commit) so the run continues on Kaggle's servers even if the browser is closed. Plots are written to `/kaggle/working/plots`, tables to `/kaggle/working/tables`, and per-run checkpoints (`.weights.h5` weights, `.json` metrics and test logits) to `/kaggle/working/checkpoints`; the summary files (`results_summary.csv`, `metrics_summary.json`, `attention_stats.json`, `split_indices.json`) are written to `/kaggle/working`, and the last cell zips the plots and tables for download. A finished run is skipped if the cell is re-run in the same session. When run outside Kaggle, the notebook automatically uses the current folder as its working directory. Each of the 16 training runs takes about 8 minutes on the T4 GPU (7.8 to 8.6 minutes, 129.1 minutes in total); the Grad-CAM, attention-map and plotting cells after the training add a few more minutes, and CPU execution is not practical.

To rebuild the report, upload `DL_Lab_Experiment_8&9_report.tex` together with the `plots/` folder to Overleaf and compile with pdfLaTeX.

## Contents

* `Experiment_8&9_code_file.ipynb`: data preparation and fixed split, checkpoint utility, the attention modules (channel/SE-style, spatial, Channel+Spatial in either order, lightweight self-attention, combined, ECA and CBAM), ResNet50 split into its four stages so an attention block can be inserted after any of them, two-phase transfer-learning training (frozen backbone, then conv5_x unfrozen), the attention-type ablation (A0 to A5 at P3), the attention-position ablation (P1, P2, P3, P4), multiple-position runs (P3+P4 and P2+P3+P4), attention order, spatial kernel size (3, 5, 7), the two additional mechanisms (ECA and CBAM), evaluation (accuracy, macro precision, recall, F1, parameters, inference time), confusion matrices, Grad-CAM, internal attention maps and the consolidated results tables.
* `DL_Lab_Experiment_8&9_report.pdf`: compiled report with the results tables, the analysis of all required plots, the attention-mechanism comparison, answers to the 35 questions, the discussion and the conclusion.
* `DL_Lab_Experiment_8&9_report.tex`: LaTeX source of the report; it compiles in Overleaf together with the `plots/` folder.
* `plots/`: all plots used in the report — sample images, accuracy and loss curves, confusion matrices (baseline, Channel+Spatial and their difference), accuracy and macro-F1 bar charts, accuracy against parameters, F1 against inference time, position/multiple-position/order/kernel summary, feature maps before and after attention, channel-attention weights, spatial-attention overlays, self-attention response, spatial attention at P2, P3 and P4, baseline against attention Grad-CAM, the qualitative grid of correct and incorrect images and the SE/ECA/CBAM Grad-CAM comparison — all as `.png`.

## Key Results

* **Baseline ResNet50 (no attention):** test accuracy 92.74%, macro F1 92.50%, 23,597,957 parameters, 4.55 ms per image
* **Channel (SE) at P3:** accuracy 92.56%, macro F1 92.44%, 23,730,117 parameters (+132,160)
* **Spatial at P3:** accuracy 91.65%, macro F1 91.51%, 23,598,056 parameters (+99)
* **Channel+Spatial at P3:** accuracy 91.11%, macro F1 90.91%, 23,730,216 parameters (+132,259)
* **Self-attention at P3:** accuracy 91.83%, macro F1 91.68%, 24,908,678 parameters (+1,310,721), 4.73 ms per image
* **Combined (Channel+Spatial+Self-attention) at P3:** accuracy 92.20%, macro F1 92.13%, 25,040,937 parameters (+1,442,980)
* **Attention position (Channel→Spatial):** P1 90.93%, P2 91.83%, P3 91.11%, P4 91.83% accuracy, a spread of five test images; P3+P4 gives 92.56% and P2+P3+P4 gives 91.47%
* **Attention order and kernel size:** Spatial→Channel gives 94.01% against 91.11% for Channel→Spatial; spatial kernels 3, 5 and 7 give 90.74%, 93.28% and 91.11%, with no steady trend
* **Additional mechanisms:** ECA 92.56% accuracy with only 5 added parameters; CBAM 91.65% with 132,259 added parameters
* **Training cost:** 7.81 minutes for the baseline against 8.03 minutes for attention at P3 and 8.36 minutes for P2+P3+P4, so even three attention blocks add only about half a minute
* **Attention diagnostics:** the spatial gate averages 0.94 to 0.98 and is close to 1 over almost the whole map, about a third of the channel gates are near 0 and the weights hardly change between images, and the learned self-attention scale is about −0.02, so the modules change the features very little
* **Class-wise behaviour:** daisy F1 is above the baseline in 13 of the 15 attention runs while tulips are below it in all 15; tulips predicted as roses rises from 13 images in the baseline to 20 with Channel+Spatial
* **Main Finding:** No attention variant clearly beat the plain pretrained ResNet50: 13 of the 15 attention runs scored below the baseline (by 1 to 11 test images), and the two runs above it, Spatial→Channel and the 5×5 kernel, are not supported by their neighbouring settings. The internal maps explain why: the spatial attention was nearly flat apart from the image border, the channel attention behaved like an almost fixed mask, the self-attention response was a single spot that did not depend on the query, and the Grad-CAM maps of the attention models matched the baseline on correct images and changed only on misclassified ones (in one case moving to the ground below the flower). Each setting was trained once, and the validation accuracy of a single run moves by about 2.8 points over its last six epochs, so accuracy gaps of about two points or less are treated as noise; the only consistent pattern is by class, where daisy improves and tulips get worse.
