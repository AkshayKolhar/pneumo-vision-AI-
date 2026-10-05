import os
import torch
import re
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


def _patient_group_id(image_path):
    stem = image_path.stem
    pneumonia_match = re.match(r"(person\d+)", stem)
    normal_match = re.match(r"(IM-\d+)", stem)
    if pneumonia_match:
        return pneumonia_match.group(1)
    if normal_match:
        return normal_match.group(1)
    return stem

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.jpeg')):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L') # Grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(data_dir='data/chest_xray', batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Separate dataset instances keep validation preprocessing independent from
    # training augmentation even though both subsets use the same source files.
    train_dataset = ChestXRayDataset(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = ChestXRayDataset(os.path.join(data_dir, 'train'), transform=val_test_transforms)
    
    # Split: 80% Train, 20% Val
    total_size = len(train_dataset)
    groups_by_label = {0: defaultdict(list), 1: defaultdict(list)}
    group_labels = {}
    for index, (image_path, label) in enumerate(train_dataset.samples):
        group_id = _patient_group_id(image_path)
        if group_id in group_labels and group_labels[group_id] != label:
            raise ValueError(
                f"Patient group {group_id!r} contains images from both classes."
            )
        group_labels[group_id] = label
        groups_by_label[label][group_id].append(index)

    random_generator = random.Random(42)
    val_indices = set()
    for label, patient_groups in groups_by_label.items():
        group_ids = sorted(patient_groups)
        if len(group_ids) < 2:
            raise ValueError(
                f"Need at least two patient groups for class {label} "
                "to create a patient-separated validation split."
            )
        random_generator.shuffle(group_ids)
        target_val_count = round(
            sum(len(indices) for indices in patient_groups.values()) * 0.2
        )
        selected_count = 0
        for group_id in group_ids:
            if selected_count >= target_val_count:
                break
            val_indices.update(patient_groups[group_id])
            selected_count += len(patient_groups[group_id])

    train_indices = sorted(set(range(total_size)) - val_indices)
    val_indices = sorted(val_indices)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Load Test set separately
    test_dataset = ChestXRayDataset(os.path.join(data_dir, 'test'), transform=val_test_transforms)
    
    # Create Loaders
    shuffle_generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
