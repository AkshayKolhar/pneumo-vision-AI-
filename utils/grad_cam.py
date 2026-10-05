from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from utils.inference import preprocess_image


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int | None = None,
) -> tuple[np.ndarray, int]:
    """Generate a normalized Grad-CAM map for the model's predicted class."""
    activations: list[torch.Tensor] = []

    def capture_activation(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activations.append(output)

    handle = target_layer.register_forward_hook(capture_activation)
    try:
        with torch.enable_grad():
            output = model(image_tensor)
            if not activations:
                raise RuntimeError("Grad-CAM target layer did not produce activations.")

            if target_class is None:
                target_class = int(output.argmax(dim=1).item())
            gradients = torch.autograd.grad(
                output[0, target_class],
                activations[0],
                retain_graph=False,
                create_graph=False,
            )[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
            cam = functional.interpolate(
                cam,
                size=image_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
            return cam.detach().cpu().numpy(), target_class
    finally:
        handle.remove()


def gradcam_images(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> tuple[Image.Image, Image.Image]:
    """Return a colored activation map and a blended preview."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between 0 and 1.")
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Heatmap must be a finite two-dimensional array.")

    base = image.convert("RGB")
    heatmap_image = Image.fromarray(
        np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    ).resize(base.size, Image.Resampling.BILINEAR)
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    color_map = matplotlib.colormaps["jet"](heatmap_array)[..., :3]
    colored = Image.fromarray((color_map * 255).astype(np.uint8), mode="RGB")
    overlay = Image.blend(base, colored, alpha)
    return colored, overlay


class GradCAM:
    """Small compatibility wrapper used by the offline visualization script."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        self.target_layer = (
            target_layer if target_layer is not None else model.backbone.layer4[-1]
        )

    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int]:
        return generate_gradcam(
            self.model,
            image_tensor,
            self.target_layer,
            target_class,
        )

    def visualize(
        self,
        image_path: str | Path,
        target_class: int | None = None,
        alpha: float = 0.5,
        cmap: str = "jet",
    ):
        del cmap
        import matplotlib.pyplot as plt

        image = Image.open(image_path).convert("L")
        device = next(self.model.parameters()).device
        tensor = preprocess_image(image).unsqueeze(0).to(device)
        heatmap, predicted_class = generate_gradcam(
            self.model,
            tensor,
            self.target_layer,
            target_class,
        )
        colored, overlay = gradcam_images(image, heatmap, alpha)

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input X-ray")
        axes[1].imshow(colored)
        axes[1].set_title("Activation map")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Model class: {predicted_class}")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        return figure, np.asarray(overlay), predicted_class
