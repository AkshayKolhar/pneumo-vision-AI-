import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from PIL import Image

from utils.dataset import get_data_loaders
from utils.grad_cam import generate_gradcam, gradcam_images
from utils.inference import exceeds_threshold, preprocess_image
from compare_models import metrics_at_threshold, select_threshold_for_recall


class InferenceTests(unittest.TestCase):
    def test_preprocessing_matches_model_input_contract(self):
        image = Image.new("RGB", (320, 240), color="white")
        tensor = preprocess_image(image)

        self.assertEqual(tuple(tensor.shape), (1, 224, 224))
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertTrue(bool((tensor == 1.0).all()))

    def test_threshold_includes_equal_score(self):
        self.assertTrue(exceeds_threshold(0.5, 0.5))
        self.assertFalse(exceeds_threshold(0.49, 0.5))

    def test_threshold_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            exceeds_threshold(1.1, 0.5)
        with self.assertRaises(ValueError):
            exceeds_threshold(0.5, -0.1)

    def test_data_split_is_reproducible_and_keeps_distinct_transforms(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ("train", "test"):
                for class_name in ("NORMAL", "PNEUMONIA"):
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True)
                    for patient in range(10):
                        for view in range(2):
                            if class_name == "NORMAL":
                                name = f"IM-{patient:04d}-{view:04d}.jpeg"
                            else:
                                name = f"person{patient}_bacteria_{view}.jpeg"
                            Image.new("L", (16, 16), color=patient).save(
                                class_dir / name
                            )

            first_train, first_val, _ = get_data_loaders(str(root), batch_size=2)
            second_train, second_val, _ = get_data_loaders(str(root), batch_size=2)

            self.assertEqual(first_train.dataset.indices, second_train.dataset.indices)
            self.assertEqual(first_val.dataset.indices, second_val.dataset.indices)
            self.assertIsNot(first_train.dataset.dataset, first_val.dataset)
            self.assertNotEqual(
                len(first_train.dataset.dataset.transform.transforms),
                len(first_val.dataset.dataset.transform.transforms),
            )
            self.assertEqual(len(first_train.dataset), 32)
            self.assertEqual(len(first_val.dataset), 8)
            self.assertEqual(
                sum(first_val.dataset.dataset.samples[index][1] == 0 for index in first_val.dataset.indices),
                4,
            )
            self.assertEqual(
                sum(first_val.dataset.dataset.samples[index][1] == 1 for index in first_val.dataset.indices),
                4,
            )

            train_groups = {
                first_train.dataset.dataset.samples[index][0].stem.rsplit("-", 1)[0]
                if first_train.dataset.dataset.samples[index][0].stem.startswith("IM-")
                else first_train.dataset.dataset.samples[index][0].stem.split("_", 1)[0]
                for index in first_train.dataset.indices
            }
            val_groups = {
                first_val.dataset.dataset.samples[index][0].stem.rsplit("-", 1)[0]
                if first_val.dataset.dataset.samples[index][0].stem.startswith("IM-")
                else first_val.dataset.dataset.samples[index][0].stem.split("_", 1)[0]
                for index in first_val.dataset.indices
            }
            self.assertFalse(train_groups & val_groups)

    def test_gradcam_returns_finite_map_and_removes_hook(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Conv2d(1, 2, kernel_size=3, padding=1)
                self.classifier = nn.Linear(2, 2)

            def forward(self, inputs):
                features = torch.relu(self.features(inputs))
                return self.classifier(features.mean(dim=(2, 3)))

        model = TinyModel().eval()
        image = torch.rand(1, 1, 16, 16)
        heatmap, target_class = generate_gradcam(model, image, model.features)

        self.assertEqual(heatmap.shape, (16, 16))
        self.assertIn(target_class, (0, 1))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertEqual(len(model.features._forward_hooks), 0)

    def test_gradcam_overlay_matches_input_dimensions(self):
        image = Image.new("L", (80, 60), color=128)
        heatmap = np.zeros((16, 16), dtype=np.float32)
        heatmap[4:12, 4:12] = 1.0

        colored, overlay = gradcam_images(image, heatmap)

        self.assertEqual(colored.size, image.size)
        self.assertEqual(overlay.size, image.size)

    def test_threshold_selection_meets_recall_constraint(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.8, 0.6, 0.9])

        selected = select_threshold_for_recall(labels, scores, 1.0)

        self.assertEqual(selected["threshold"], 0.6)
        self.assertEqual(selected["pneumonia_recall"], 1.0)
        self.assertEqual(selected["normal_specificity"], 0.5)

    def test_threshold_metrics_report_false_positives_and_misses(self):
        result = metrics_at_threshold(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.2, 0.8, 0.4, 0.9]),
            0.5,
        )

        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["pneumonia_recall"], 0.5)
        self.assertEqual(result["normal_specificity"], 0.5)


if __name__ == "__main__":
    unittest.main()
