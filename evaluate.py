import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    recall_score,
    precision_score,
    f1_score
)
from utils.dataset import get_data_loaders
from utils.inference import load_model

def evaluate_model(model_path='models/best_pneumonia_model.pth', 
                   data_dir='data/chest_xray',
                   batch_size=32):
    """
    Evaluate the trained model on test set with focus on recall
    
    Args:
        model_path: Path to saved model
        data_dir: Path to dataset
        batch_size: Batch size for evaluation
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(model_path, device)

    print(f"Model loaded from {model_path}")
    
    # Get test loader
    _, _, test_loader = get_data_loaders(data_dir, batch_size)
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    accuracy = float((all_preds == all_labels).mean())
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['Normal', 'Pneumonia']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print(f"Confusion Matrix:\n{cm}")

    true_normal, false_pneumonia = cm[0]
    specificity = (
        float(true_normal / (true_normal + false_pneumonia))
        if true_normal + false_pneumonia
        else 0.0
    )
    model_digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    report = {
        "model_sha256": model_digest,
        "samples": int(len(all_labels)),
        "accuracy": accuracy,
        "pneumonia_recall": float(recall),
        "pneumonia_precision": float(precision),
        "pneumonia_f1": float(f1),
        "normal_specificity": specificity,
        "confusion_matrix": cm.tolist(),
    }
    report_path = Path("evaluation_results.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metrics saved to '{report_path}'")
    
    # Plot results
    plot_evaluation_results(all_labels, all_preds, all_probs, cm)
    
    return recall, precision, f1

def plot_evaluation_results(labels, preds, probs, cm):
    """Plot evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Confusion Matrix
    im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix', fontsize=14)
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(['Normal', 'Pneumonia'])
    axes[0, 0].set_yticklabels(['Normal', 'Pneumonia'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, str(cm[i, j]),
                           ha="center", va="center",
                           color="white" if cm[i, j] > cm.max()/2 else "black",
                           fontsize=14)
    plt.colorbar(im, ax=axes[0, 0])
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('Receiver Operating Characteristic', fontsize=14)
    axes[0, 1].legend(loc="lower right", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
    axes[1, 0].plot(recall_vals, precision_vals, color='green', lw=2)
    axes[1, 0].set_xlabel('Recall', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_title('Precision-Recall Curve', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of predictions
    normal_probs = probs[labels == 0]
    pneumonia_probs = probs[labels == 1]
    axes[1, 1].hist(normal_probs, bins=30, alpha=0.5, label='Normal', color='blue')
    axes[1, 1].hist(pneumonia_probs, bins=30, alpha=0.5, label='Pneumonia', color='red')
    axes[1, 1].set_xlabel('Probability of Pneumonia', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Prediction Probability Distribution', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nEvaluation plots saved to 'evaluation_results.png'")

if __name__ == '__main__':
    evaluate_model()