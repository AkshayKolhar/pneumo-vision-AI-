import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.pneumonia_model import PneumoniaDetector
from utils.dataset import get_data_loaders


EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
SEED = 42
PATIENCE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("models/challenger_pneumonia_model.pth")
HISTORY_PATH = Path("models/challenger_training_history.json")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validation_metrics(model, loader, criterion, use_amp):
    model.eval()
    total_loss = 0.0
    probabilities = []
    labels = []

    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * len(batch_labels)
            probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    predictions = np.asarray(probabilities) >= 0.5
    labels_array = np.asarray(labels)
    average_precision = average_precision_score(labels_array, probabilities)
    recall = recall_score(labels_array, predictions, zero_division=0)
    return (
        total_loss / len(loader.dataset),
        float(average_precision),
        float(recall),
        probabilities,
        labels,
    )


def train():
    _seed_everything(SEED)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    train_loader, val_loader, _ = get_data_loaders(batch_size=BATCH_SIZE)
    model = PneumoniaDetector(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_average_precision = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
        )
        for images, labels in progress:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_ap, val_recall, _, _ = _validation_metrics(
            model,
            val_loader,
            criterion,
            use_amp,
        )
        scheduler.step(val_ap)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_average_precision": val_ap,
            "validation_recall_at_0_5": val_recall,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"Val AP: {val_ap:.4f} | Val recall @ 0.50: {val_recall:.4f}"
        )

        if val_ap > best_average_precision:
            best_average_precision = val_ap
            epochs_without_improvement = 0
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_average_precision": val_ap,
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "class_weights": None,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> Saved challenger checkpoint: {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without AP improvement.")
                break

        HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best validation average precision: {best_average_precision:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")


if __name__ == "__main__":
    train()
