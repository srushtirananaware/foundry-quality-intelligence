from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# --------------------------------------------------
# 1. Dataset location
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "casting_data" / "casting_data"


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# --------------------------------------------------
# 3. Load training dataset
# --------------------------------------------------

full_train_dataset = datasets.ImageFolder(
    root=DATASET_PATH / "train",
    transform=transform
)


# --------------------------------------------------
# 4. Split training data into train + validation
# --------------------------------------------------

train_size = int(0.8 * len(full_train_dataset))
validation_size = len(full_train_dataset) - train_size

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)


# --------------------------------------------------
# 5. Load test dataset separately
# --------------------------------------------------

test_dataset = datasets.ImageFolder(
    root=DATASET_PATH / "test",
    transform=transform
)


# --------------------------------------------------
# 6. Create DataLoaders
# --------------------------------------------------

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# 7. Dataset information
# --------------------------------------------------

if __name__ == "__main__":

    print("Classes:", full_train_dataset.classes)

    print("\nTraining images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("\nBatch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("First 10 labels:", labels[:10])