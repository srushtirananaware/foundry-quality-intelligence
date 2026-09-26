from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "casting_data"
    / "casting_data"
)


# Preprocessing for pretrained ResNet18
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


full_train_dataset = datasets.ImageFolder(
    root=DATASET_PATH / "train",
    transform=resnet_transform
)


train_size = int(0.8 * len(full_train_dataset))
validation_size = len(full_train_dataset) - train_size


train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)


test_dataset = datasets.ImageFolder(
    root=DATASET_PATH / "test",
    transform=resnet_transform
)


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


if __name__ == "__main__":

    print("Classes:", full_train_dataset.classes)

    print("\nTraining images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("\nBatch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("First 10 labels:", labels[:10])