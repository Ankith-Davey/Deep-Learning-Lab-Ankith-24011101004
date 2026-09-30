# Experiment 7 — End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders

CS3807 Deep Learning Laboratory, AY 2026-27

## Dataset

MNIST handwritten digits (28×28×1 grayscale), loaded directly through `keras.datasets.mnist`. Pixel values scaled from [0, 255] to [0, 1]. A 10,000-image training subset and a 2,000-image test subset were drawn from MNIST with a fixed random seed (109) and reused across every model so all comparisons are controlled. Labels are not used as reconstruction targets (the input image is the target); they are used only for colouring the VAE latent-space plots and for selecting digit centroids in the interpolation exercise. Fully connected models take flattened 784-dimensional inputs; convolutional and variational models keep the 28×28×1 spatial structure. For the denoising study, Gaussian noise (σ = 0.1, 0.2, 0.3) and salt-and-pepper noise (p = 0.05, 0.10, 0.20) were applied to the inputs, with the clean image kept as the target.

## Dependencies

tensorflow, keras, numpy, pandas, matplotlib, scikit-image (SSIM), scikit-learn (PCA projection of the 8-dimensional VAE latent space)

## How to run

Open `Experiment_7.ipynb` in Kaggle on a GPU runtime (T4 GPU). Run cells sequentially; MNIST is downloaded automatically and no dataset needs to be attached. Plots are written to `/kaggle/working/plots` and metrics (`consolidated_results.csv`, `vae_specific_results.csv`, `ALL_METRICS_SUMMARY.json`) and weights to `/kaggle/working/checkpoints`; the last cell zips both for download. Running locally requires changing these two paths in the setup cell. Individual model training takes roughly 11–22 seconds on the T4 GPU (see the consolidated table in the report); the latent-dimension sweeps, six denoising models and the β-VAE ablation add to the total, and CPU execution will be noticeably slower.

## Contents

* `Experiment_7.ipynb`: data preparation and metric utilities (MSE, MAE, SSIM), fully connected autoencoder, convolutional autoencoder (UpSampling2D and Conv2DTranspose decoders), denoising convolutional autoencoder (six noise configurations), variational autoencoder (latent dimensions 2 and 8), latent-space visualization, random generation, latent interpolation, reconstruction-error analysis, latent-dimension sweeps for the FC-AE and CAE, β-VAE ablation, and the consolidated results table.
* `Experiment_7_Report.pdf`: compiled report with analysis of all required plots, the eight additional exercises, and answers to the 25 discussion questions.
* `Experiment 7 Plots/`: all generated plots — FC-AE reconstructions and loss curves, FC-AE vs CAE comparison, UpSampling vs Conv2DTranspose, clean/noisy/denoised grids, noise level vs quality, Gaussian vs salt-and-pepper, VAE latent space, VAE generated samples (25 and 100), latent interpolations (3→8 and 0→1 centroid), VAE loss curves, reconstruction-error histogram, five highest-error images, FC-AE and CAE latent-capacity sweeps, VAE latent 2 vs 8, and β-VAE latent spaces — exported as `.png`.

## Key Results

* **FC-AE (latent 16):** test MSE 0.02252, MAE 0.05972, SSIM 0.7452, 211,040 parameters, 11.5s training time
* **CAE (UpSampling):** MSE 0.00246, MAE 0.01502, SSIM 0.9751, 74,497 parameters, 19.5s training time
* **CAE (Conv2DTranspose):** MSE 0.00224, MAE 0.01402, SSIM 0.9779, 83,745 parameters, 19.1s training time
* **Denoising CAE:** Gaussian σ = 0.2 gives MSE 0.00438, SSIM 0.9489; salt-and-pepper p = 0.10 gives MSE 0.00458, SSIM 0.9497; SSIM falls only gradually from 0.9666 (σ = 0.1) to 0.9236 (σ = 0.3)
* **VAE (latent 2):** MSE 0.04475, SSIM 0.5007, reconstruction loss 150.06, KL loss 5.44, 284,933 parameters
* **VAE (latent 8):** MSE 0.02135, SSIM 0.7737, reconstruction loss 101.72, KL loss 16.51, 304,529 parameters
* **β-VAE (latent 2):** reconstruction loss rises from 147.27 to 161.27 and KL loss falls from 6.78 to 3.04 as β goes from 0.5 to 5.0
* **Main Finding:** Preserving spatial structure mattered far more than parameter count: the CAE cut MSE by roughly 9× and raised SSIM by 0.23 over the FC-AE using about a third of the parameters, and stayed accurate even with only 4 bottleneck channels (SSIM 0.9473) while the FC-AE degraded sharply at small latent sizes; the denoising CAE recovered digit structure under both noise types with only a gradual drop in SSIM; and the VAE reconstructed worse than every deterministic model by design, with only digits 0 and 1 separating cleanly in its 2D latent space while the rest overlapped centrally, which also explained why random samples from the crowded centre were ambiguous and why interpolation between arbitrary test points barely changed while interpolation between the 0 and 1 centroids gave a clear gradual morph.
