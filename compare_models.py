import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from utils.dataset import get_data_loaders
from utils.inference import load_model


DATA_DIR = Path("data/chest_xray")
BASELINE_PATH = Path("models/best_pneumonia_model.pth")
CHALLENGER_PATH = Path("models/challenger_pneumonia_model.pth")
REPORT_PATH = Path("models/model_comparison.json")
MIN_VALIDATION_RECALL = 0.98
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_scores(model, loader):
    labels = []
    scores = []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            outputs = model(images.to(DEVICE))
            scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(labels), np.asarray(scores)


def metrics_at_threshold(labels, scores, threshold):
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "samples": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "pneumonia_recall": float(recall),
        "pneumonia_precision": float(precision),
        "pneumonia_f1": float(f1),
        "normal_specificity": float(specificity),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold_for_recall(labels, scores, minimum_recall):
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    feasible = [
        metrics_at_threshold(labels, scores, threshold)
        for threshold in candidates
        if metrics_at_threshold(labels, scores, threshold)["pneumonia_recall"]
        >= minimum_recall
    ]
    if not feasible:
        raise ValueError(
            f"No threshold achieved validation recall of {minimum_recall:.1%}."
        )
    return max(
        feasible,
        key=lambda result: (
            result["normal_specificity"],
            result["threshold"],
        ),
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_models():
    for checkpoint in (BASELINE_PATH, CHALLENGER_PATH):
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Required checkpoint is missing: {checkpoint}. "
                "Run python train.py before comparing."
            )

    print(f"Using device: {DEVICE}")
    _, validation_loader, test_loader = get_data_loaders(
        str(DATA_DIR),
        batch_size=32,
    )

    results = {}
    for name, path in (
        ("baseline", BASELINE_PATH),
        ("challenger", CHALLENGER_PATH),
    ):
        print(f"\nLoading {name}: {path}")
        model = load_model(path, DEVICE)
        test_labels, test_scores = collect_scores(model, test_loader)
        results[name] = {"checkpoint_sha256": sha256(path)}

        if name == "baseline":
            results[name]["threshold_policy"] = (
                "Fixed at 0.50; the original training split is unknown, so "
                "the current validation split is not used to tune this checkpoint."
            )
            results[name]["test_at_0_50"] = metrics_at_threshold(
                test_labels,
                test_scores,
                0.5,
            )
            print("Baseline test metrics at fixed threshold 0.50:")
            print(json.dumps(results[name]["test_at_0_50"], indent=2))
        else:
            validation_labels, validation_scores = collect_scores(
                model,
                validation_loader,
            )
            selected = select_threshold_for_recall(
                validation_labels,
                validation_scores,
                MIN_VALIDATION_RECALL,
            )
            results[name]["validation_at_0_50"] = metrics_at_threshold(
                validation_labels,
                validation_scores,
                0.5,
            )
            results[name]["threshold_selected_on_validation"] = selected
            results[name]["test_at_0_50"] = metrics_at_threshold(
                test_labels,
                test_scores,
                0.5,
            )
            results[name]["test_at_selected_threshold"] = metrics_at_threshold(
                test_labels,
                test_scores,
                selected["threshold"],
            )
            print("Challenger validation metrics at 0.50:")
            print(json.dumps(results[name]["validation_at_0_50"], indent=2))
            print(
                f"Validation-selected threshold for recall >= "
                f"{MIN_VALIDATION_RECALL:.0%}: {selected['threshold']:.6f}"
            )
            print("Challenger test metrics at the validation-selected threshold:")
            print(json.dumps(results[name]["test_at_selected_threshold"], indent=2))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "dataset": str(DATA_DIR),
        "selection_split": "stratified patient-group validation split from train/",
        "test_split": "test/",
        "minimum_validation_recall": MIN_VALIDATION_RECALL,
        "threshold_selection_note": (
            "Exploratory operating point only; selected on validation data, "
            "not a clinical recommendation. Baseline remains at the fixed 0.50 "
            "threshold because its training split is unknown."
        ),
        "models": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nComparison report saved to {REPORT_PATH}")


if __name__ == "__main__":
    compare_models()
