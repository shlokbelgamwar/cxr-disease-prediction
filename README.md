# CXR Disease Prediction

Multi-label NIH Chest X-ray classification using PyTorch and ResNet50.

## Notebooks

- `notebooks/cxr_disease_prediction_original.ipynb`: original Kaggle notebook with its historical saved outputs.
- `notebooks/cxr_disease_prediction_improved.ipynb`: revised BCE baseline with patient-level stratification and overlap checks, 14 disease targets, validation-derived thresholds, per-disease reports, and explicit failures for missing or corrupt images.

[Original Kaggle notebook](https://www.kaggle.com/code/shlokbelgamwar/cxr-disease-prediction/edit) | [Improved Kaggle notebook](https://www.kaggle.com/code/shlokbelgamwar/cxr-disease-prediction-improved/edit)

These are source snapshots prepared before the current training run. Live epoch outputs, trained checkpoints, and final metrics are not included. Python syntax was checked; this repository does not claim improved measured performance.

## Run on Kaggle

1. Import the improved notebook into Kaggle. Its metadata references the NIH Chest X-rays dataset.
2. Confirm that NIH Chest X-rays is attached, GPU acceleration is enabled, and internet access is available for package/model downloads.
3. Run the cells in order. The dataset root defaults to `/kaggle/input/data`; automatic metadata discovery handles a different mount name when exactly one metadata CSV is found.
4. The default `LOSS_MODE = "bce"` trains an unweighted baseline. Use `"weighted_bce"` in a separate run to compare train-only negative/positive weighting.
5. Compare validation results when selecting experiments. Freeze thresholds before evaluating the official test set. Do not select models or losses using test performance.

Artifacts are written to `/kaggle/working/resnet50_<loss_mode>/`, including the best model, patient split manifests, split prevalence, training class weights, thresholds, configuration, and final test reports/predictions. Copy completed run artifacts separately if they need to be preserved outside Kaggle.

The improved notebook treats No Finding as an all-zero vector across the 14 disease targets. No Finding is a dataset annotation, not a guarantee of health. Changing the split and label count means results are not directly comparable to the original aggregate metrics.

## Related theory paper

**Democratizing Medical Imaging: Knowledge Distillation for Lightweight Multi-label AI Models**  
Shlok Belgamwar, Madhuri Wakode, and Geetanjali Kale (2026). *ICT for Intelligent Systems*, ICTIS 2025, Lecture Notes in Networks and Systems, volume 1517, pp. 215–226. Springer, Singapore.

[Publisher chapter](https://link.springer.com/chapter/10.1007/978-981-96-8895-1_19) | [DOI](https://doi.org/10.1007/978-981-96-8895-1_19)

The paper provides the conceptual context for lightweight medical-imaging models through knowledge distillation. The notebooks currently implement a ResNet50 classification baseline; they do not implement the paper's complete distillation framework or establish its empirical results. Citation metadata is in `CITATION.bib`. The full-text PDF is not bundled; publisher access may require institutional access or purchase.

## Dependencies

Python 3.10 or later in a compatible PyTorch environment. See `requirements.txt` for direct dependencies; versions are not a lockfile. The improved notebook uses the modern `torch.amp` API and requires PyTorch 2.3 or later. Kaggle runtime metadata is retained in the notebook.

## Scope

Research and educational use. This code is not validated for clinical diagnosis. Dataset images, credentials, and trained model binaries are excluded from source control.
