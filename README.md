# WM-811K Wafer Map Classification

Reproducible experiments for nine-class wafer-map defect classification. The project uses
lot-aware data splitting, a shared training engine, and configuration-driven comparison of
Custom CNN, ResNet18, ConvNeXt-Tiny, and Swin-Tiny.

## Final result

ResNet18 was selected using validation Macro F1 and evaluated once on the frozen test split.

| Metric | Result |
|---|---:|
| Validation Macro F1 | 0.8740 |
| Test Macro F1 | **0.8277** |
| Test balanced accuracy | 0.9014 |
| Test accuracy | 0.9671 |

The split is grouped by `lotName`, so wafers from a production lot cannot leak across train,
validation, and test. See [FINAL_REPORT.md](FINAL_REPORT.md) for model comparison, per-class
metrics, high-confidence errors, and Grad-CAM analysis.

## Model comparison

| Model | Validation Macro F1 |
|---|---:|
| ResNet18 | **0.8740** |
| Swin-Tiny stable | 0.8633 |
| Custom CNN | 0.8398 |
| ConvNeXt-Tiny | 0.7663 |
| Random Forest | 0.7284 |
| Majority baseline | 0.1022 |

## Environment

```powershell
conda activate py39
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected GPU environment: PyTorch `2.7.1+cu128` and CUDA availability `True`.

## Commands

List registered models:

```powershell
python main.py models
```

Audit the raw pickle dataset:

```powershell
python main.py inspect --data data/LSWMD.pkl
```

Create fixed-size memory-mapped arrays and lot-aware splits:

```powershell
python main.py preprocess --config configs/data.yaml
```

Generate train-only exploratory data analysis figures:

```powershell
python main.py eda --config configs/data.yaml
```

Train the interpretable spatial-feature Random Forest baseline:

```powershell
python main.py baseline --config configs/random_forest.yaml
```

This command also records a majority-class baseline. Both baselines are evaluated on validation
only; the test split remains untouched during model comparison.

Train a model:

```powershell
python main.py train --config configs/custom_cnn.yaml
```

Inspect per-class validation metrics for the best checkpoint:

```powershell
python main.py validate `
  --config configs/custom_cnn.yaml `
  --checkpoint outputs/checkpoints/custom_cnn_64_best.pt
```

Training uses only the training and validation splits. Compare models using validation Macro F1,
then evaluate the selected final checkpoint on the test split once.

If the original Swin-Tiny run collapses under extreme inverse-frequency weights, run the preserved
stability experiment with square-root/clipped class weights, BF16, warmup, and gradient clipping:

```powershell
python main.py train --config configs/swin_tiny_stable.yaml
```

Evaluate a saved checkpoint:

```powershell
python main.py evaluate `
  --config configs/custom_cnn.yaml `
  --checkpoint outputs/checkpoints/custom_cnn_64_best.pt
```

Build the frozen final comparison, error analysis, and Grad-CAM report:

```powershell
python main.py report --config configs/report.yaml
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

## Data representation

The processed cache stores one `uint8` wafer map per sample:

- `0`: outside the wafer
- `1`: valid/good die
- `2`: failed die

Before resizing, non-square maps are centered and zero-padded to a square so defect geometry is
not stretched. The Dataset then converts each map at runtime into two float channels: valid-wafer
mask and failed-die mask. Rotation and flip augmentation are applied to training samples only.

Generated arrays, checkpoints, metrics, logs, and figures are intentionally ignored by Git.
