from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.pneumonia_model import PneumoniaDetector


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def load_model(model_path: str | Path, device: torch.device) -> PneumoniaDetector:
    """Load a plain state dict or a checkpoint containing ``model_state_dict``."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    model = PneumoniaDetector(pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the grayscale resize and normalization used during evaluation."""
    return IMAGE_TRANSFORM(image.convert("L"))


def pneumonia_probability(
    model: PneumoniaDetector,
    image: Image.Image,
    device: torch.device,
) -> float:
    """Return the model's pneumonia-class softmax score for one image."""
    tensor = preprocess_image(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)
    return float(probabilities[0, 1].cpu())


def exceeds_threshold(probability: float, threshold: float) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0 and 1.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    return probability >= threshold
