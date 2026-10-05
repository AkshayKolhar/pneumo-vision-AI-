import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from compare_models import DEVICE, metrics_at_threshold
from utils.dataset import ChestXRayDataset, _patient_group_id, get_data_loaders
from utils.inference import load_model


DATA_DIR = Path("data/chest_xray")
MODEL_PATH = Path("models/challenger_pneumonia_model.pth")
COMPARISON_PATH = Path("models/model_comparison.json")
REPORT_PATH = Path("models/challenger_error_analysis.json")
CSV_PATH = Path("models/challenger_test_errors.csv")
VALIDATION_SWEEP_PATH = Path("models/challenger_validation_threshold_sweep.csv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_scores(model, loader):
    labels = []
    scores = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            outputs = model(images.to(DEVICE))
            scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(labels), np.asarray(scores)


def filename_group_audit(train: ChestXRayDataset, test: ChestXRayDataset):
    train_groups = defaultdict(set)
    test_groups = defaultdict(set)
    for path, label in train.samples:
        train_groups[label].add(_patient_group_id(path))
    for path, label in test.samples:
        test_groups[label].add(_patient_group_id(path))

    overlap = {}
    for label, class_name in ((0, "NORMAL"), (1, "PNEUMONIA")):
        shared_groups = train_groups[label] & test_groups[label]
        overlap[class_name] = {
            "train_group_prefix_count": len(train_groups[label]),
            "test_group_prefix_count": len(test_groups[label]),
            "shared_filename_group_prefix_count": len(shared_groups),
            "examples": sorted(shared_groups)[:20],
        }
    return overlap


def exact_duplicate_count(train: ChestXRayDataset, test: ChestXRayDataset) -> int:
    train_hashes = {
        hashlib.sha256(path.read_bytes()).digest()
        for path, _ in train.samples
    }
    return sum(
        hashlib.sha256(path.read_bytes()).digest() in train_hashes
        for path, _ in test.samples
    )


def analyze_errors():
    for path in (MODEL_PATH, COMPARISON_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis input is missing: {path}")

    checkpoint_digest = sha256_file(MODEL_PATH)
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    model_report = comparison.get("models", {}).get("challenger")
    if model_report is None or model_report.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError(
            "Comparison report does not match the challenger checkpoint. "
            "Run python compare_models.py again before error analysis."
        )

    threshold = model_report["threshold_selected_on_validation"]["threshold"]
    _, validation_loader, test_loader = get_data_loaders(
        str(DATA_DIR),
        batch_size=32,
    )
    model = load_model(MODEL_PATH, DEVICE)

    validation_labels, validation_scores = collect_scores(model, validation_loader)
    labels, scores = collect_scores(model, test_loader)

    dataset = test_loader.dataset
    if len(dataset.samples) != len(scores):
        raise RuntimeError(
            "Test dataset order no longer matches predictions; refusing to "
            "associate scores with image paths."
        )

    rows = []
    for (image_path, label), score in zip(dataset.samples, scores):
        prediction = int(score >= threshold)
        if prediction == label:
            continue
        rows.append(
            {
                "file": image_path.name,
                "path": str(image_path),
                "actual": "PNEUMONIA" if label else "NORMAL",
                "predicted": "PNEUMONIA" if prediction else "NORMAL",
                "pneumonia_score": float(score),
                "threshold": float(threshold),
                "error_type": (
                    "false_positive" if prediction == 1 else "false_negative"
                ),
                "distance_from_threshold": float(abs(score - threshold)),
            }
        )

    rows.sort(key=lambda row: (row["error_type"], row["distance_from_threshold"]))
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "file",
            "path",
            "actual",
            "predicted",
            "pneumonia_score",
            "threshold",
            "error_type",
            "distance_from_threshold",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    train_dataset = ChestXRayDataset(DATA_DIR / "train")
    test_dataset = test_loader.dataset
    thresholds = np.unique(
        np.concatenate(
            (
                np.round(np.arange(0.1, 0.91, 0.05), decimals=8),
                np.asarray([0.5, threshold]),
            )
        )
    )
    validation_sweep = [
        metrics_at_threshold(
            validation_labels,
            validation_scores,
            float(validation_threshold),
        )
        for validation_threshold in thresholds
    ]
    with VALIDATION_SWEEP_PATH.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "threshold",
            "samples",
            "accuracy",
            "pneumonia_recall",
            "pneumonia_precision",
            "pneumonia_f1",
            "normal_specificity",
            "confusion_matrix",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in validation_sweep:
            writer.writerow(
                {
                    **result,
                    "confusion_matrix": json.dumps(result["confusion_matrix"]),
                }
            )

    report = {
        "checkpoint_sha256": checkpoint_digest,
        "test_sample_count": len(labels),
        "threshold": float(threshold),
        "threshold_source": "validation split; not optimized on test data",
        "test_metrics": metrics_at_threshold(
            labels,
            scores,
            threshold,
        ),
        "validation_threshold_sweep": validation_sweep,
        "validation_threshold_sweep_csv": str(VALIDATION_SWEEP_PATH),
        "error_counts": {
            "false_positives": sum(row["error_type"] == "false_positive" for row in rows),
            "false_negatives": sum(row["error_type"] == "false_negative" for row in rows),
        },
        "errors_csv": str(CSV_PATH),
        "filename_group_overlap_audit": filename_group_audit(
            train_dataset,
            test_dataset,
        ),
        "exact_train_test_image_duplicates": exact_duplicate_count(
            train_dataset,
            test_dataset,
        ),
        "split_independence_caveat": (
            "Filename group prefixes are not verified patient identifiers. "
            "Some pneumonia prefixes occur in both train/ and test/. Exact "
            "byte-identical file duplicates are counted separately. Without "
            "trusted patient metadata, patient-level train/test independence "
            "cannot be confirmed."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Using device: {DEVICE}")
    print(f"Threshold selected on validation: {threshold:.6f}")
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"False positives: {report['error_counts']['false_positives']}")
    print(f"False negatives: {report['error_counts']['false_negatives']}")
    print(f"Misclassified samples: {CSV_PATH}")
    print(f"Audit report: {REPORT_PATH}")


if __name__ == "__main__":
    analyze_errors()
