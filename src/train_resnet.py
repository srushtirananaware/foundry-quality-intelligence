import torch
import torch.nn as nn
import torch.optim as optim

from resnet_dataset import train_loader, validation_loader
from models.resnet18_model import CastingResNet18


# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Create ResNet18
model = CastingResNet18().to(device)


# Freeze the pretrained ResNet layers
for param in model.model.parameters():
    param.requires_grad = False


# Only train our new final classifier
for param in model.model.fc.parameters():
    param.requires_grad = True


criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.model.fc.parameters(),
    lr=0.001
)


EPOCHS = 5

best_validation_accuracy = 0.0


for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            predictions = (
                torch.sigmoid(outputs) >= 0.5
            ).float()

            validation_correct += (
                predictions == labels
            ).sum().item()

            validation_total += labels.size(0)

    validation_loss /= len(validation_loader)

    validation_accuracy = (
        validation_correct / validation_total
    )


    # -------------------------
    # Save best model
    # -------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            "models/resnet18_best.pth"
        )

        print("  ✓ Best ResNet model saved!")


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )