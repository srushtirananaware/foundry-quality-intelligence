import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from resnet_dataset import test_loader
from models.resnet18_model import CastingResNet18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load ResNet18
model = CastingResNet18().to(device)

model.load_state_dict(
    torch.load(
        "models/resnet18_best.pth",
        map_location=device
    )
)

model.eval()


all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).int()

        all_predictions.extend(
            predictions.cpu().numpy().flatten()
        )

        all_labels.extend(
            labels.numpy()
        )


# Calculate metrics
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


print("\n" + "=" * 50)
print("RESNET18 — TEST RESULTS")
print("=" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Defective", "OK"],
        zero_division=0
    )
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)