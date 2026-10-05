<<<<<<< HEAD
# pneumo-vision-AI-
=======
# LungLens

LungLens is a hackathon-ready educational demo for chest X-ray image
classification using a fine-tuned ResNet-50. It includes a Streamlit interface
for image uploads, a repeatable training/validation split, test evaluation, and
a model card with limitations.

> **Research prototype only.** This project is not a medical device and is not
> validated for clinical use. It must not be used to diagnose, treat, or rule
> out disease.

## What the demo does

- Accepts JPG, JPEG, or PNG images up to 10 MB.
- Converts an image to grayscale, resizes it to 224 × 224, and applies the
  normalization used during training.
- Shows the model's pneumonia-class softmax score and a user-adjustable display
  threshold.
- Can show an illustrative Grad-CAM activation map for the model's predicted
  class; it is not a clinical explanation or disease-localization tool.
- Keeps uploaded images in memory; the app does not intentionally save them.
- Includes an explicit model card and performance limitations.

The score is not a calibrated probability of disease. Changing the threshold
changes only the on-screen flag; it does not retrain or calibrate the model.

## Quick start

Python 3.10 or later is recommended. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

For this RTX 3050/Windows setup, install the matching CUDA 13 wheels after the
base requirements to enable GPU inference and training:

```powershell
python -m pip install --force-reinstall --no-deps `
  torch==2.14.1+cu130 torchvision==0.29.1+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
```

The app expects `models/best_pneumonia_model.pth`. This checkpoint is local and
is not included in source control. If it is absent, train the model first.
Uploads are limited to 10 MB by the included Streamlit server configuration.

## Dataset layout

Place the Kaggle Chest X-Ray dataset under `data/chest_xray/` with this layout:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/                  # optional; current scripts split train/ into train/val
```

The loader reads `.jpeg` files from `train/` and `test/`. Keep the dataset
private/local as appropriate for its license and provenance.

## Train and evaluate

```powershell
python train.py
python compare_models.py
python analyze_errors.py
```

Training runs for up to 15 epochs by default, selects CUDA when available, and
saves the best checkpoint to `models/challenger_pneumonia_model.pth`. It uses
unweighted cross-entropy, mixed precision on CUDA, early stopping by validation
average precision, and saves training history separately. Validation is a
deterministic, approximately 80/20, class-stratified patient-group split of
`train/` (seed 42); related images are kept on one side of the split. This
challenger run does not overwrite the original checkpoint.

The comparison script evaluates the untouched `test/` split. For the new
challenger, it selects an exploratory threshold on validation data to retain at
least 98% validation recall, then reports test metrics both at that threshold
and at 0.50. The 98% target is an experiment setting, not a clinical
recommendation. The original model's training split is unknown, so its
threshold remains fixed at 0.50. Test results are written to
`models/model_comparison.json`; do not use test metrics to tune a threshold.
`analyze_errors.py` exports challenger false positives and false negatives to
`models/challenger_test_errors.csv` and writes a split audit to
`models/challenger_error_analysis.json`. It also exports a threshold sweep
computed on validation only to `models/challenger_validation_threshold_sweep.csv`.
The current audit found 169 shared pneumonia filename prefixes across train/test
and zero byte-identical files. Filename prefixes are not verified patient
identifiers; patient-level train/test independence cannot be confirmed without
trusted metadata.

`python evaluate.py` remains available to reproduce the original checkpoint's
standard 0.50-threshold test report and plots. It saves
`evaluation_results.png` plus `evaluation_results.json`.

## Current comparison snapshot

Both checkpoints were evaluated on the same 619-image test set. The challenger
results below use the threshold selected on the separate validation split:

| Metric | Original model (0.50) | Challenger (validation threshold 0.341) |
|---|---:|---:|
| Accuracy | 71.1% | 86.3% |
| Pneumonia recall | 100.0% | 99.0% |
| Pneumonia precision | 68.3% | 82.5% |
| Normal specificity | 23.5% | 65.4% |
| Normal incorrectly flagged | 179 / 234 | 81 / 234 |
| Pneumonia missed | 0 / 385 | 4 / 385 |

The challenger substantially reduces false positives on this split, but still
incorrectly flags 81 normal images and misses 4 pneumonia images. These results
are descriptive only: the audit found 169 pneumonia filename-group prefixes
shared between train and test (with zero exact byte-identical files), and
filename prefixes are not verified patient identifiers. Until provenance can
confirm patient-level independence, these test metrics may be optimistic and
must not be treated as reliable external validation or generalized to clinical
populations. The app defaults to the challenger and its validation-selected
display threshold when the matching comparison report is available; otherwise
it falls back to the original checkpoint.

## Project structure

```text
app.py                    Streamlit demo
train.py                  Model training
evaluate.py               Held-out test evaluation and plots
compare_models.py         Baseline/challenger comparison and threshold selection
analyze_errors.py         Test error examples and split-overlap audit
models/pneumonia_model.py ResNet-50 classifier
utils/dataset.py          Dataset and data-loader construction
utils/inference.py        Shared model loading and preprocessing
tests/                    Lightweight inference contract tests
```

## Tests

```powershell
python -m unittest discover -s tests
```

## Responsible-use notes

- Do not upload identifiable patient images to a public/shared demo.
- Do not use predictions to make care decisions or delay professional review.
- A score below threshold cannot rule out pneumonia.
- The model has not been externally validated, prospectively evaluated, or
  reviewed for regulatory compliance.
- Before any real-world use, obtain appropriate clinical, privacy, security,
  fairness, and regulatory review and evaluate on representative external data.
>>>>>>> 0acd8fe (it is completed)
