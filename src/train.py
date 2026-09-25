import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, validation_loader
from models.baseline_cnn import CastingCNN


# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Create model
model = CastingCNN().to(device)

# 3. Loss function
criterion = nn.BCEWithLogitsLoss()

# 4. Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5. Training settings
EPOCHS = 5
best_validation_accuracy = 0.0

# 6. Training loop
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        # BCEWithLogitsLoss expects floating-point labels
        labels = labels.float().unsqueeze(1).to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Track statistics
        running_loss += loss.item()

        predictions = (torch.sigmoid(outputs) >= 0.5).float()

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # Validation
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

            predictions = (torch.sigmoid(outputs) >= 0.5).float()

            validation_correct += (
                (predictions == labels).sum().item()
            )

            validation_total += labels.size(0)

    validation_loss /= len(validation_loader)
validation_accuracy = validation_correct / validation_total


# Save the model if validation accuracy improves
if validation_accuracy > best_validation_accuracy:

    best_validation_accuracy = validation_accuracy

    torch.save(
        model.state_dict(),
        "../models/baseline_cnn_best.pth"
    )

    print("  ✓ Best model saved!")


print(
    f"Epoch [{epoch + 1}/{EPOCHS}] "
    f"Train Loss: {train_loss:.4f} | "
    f"Train Accuracy: {train_accuracy:.4f} | "
    f"Validation Loss: {validation_loss:.4f} | "
    f"Validation Accuracy: {validation_accuracy:.4f}"
)