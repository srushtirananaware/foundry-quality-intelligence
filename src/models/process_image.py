from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from models.resnet18_model import CastingResNet18


MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"


# Must match the preprocessing used during ResNet18 training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    model = CastingResNet18()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


def predict_image(image_path):
    """
    Predict whether a casting image is defective or OK.
    """

    model = load_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        defective_probability = torch.sigmoid(output).item()

    ok_probability = 1 - defective_probability

    if defective_probability >= 0.5:
        prediction = "Defective"
        confidence = defective_probability
    else:
        prediction = "OK"
        confidence = ok_probability

    return {
        "prediction": prediction,
        "confidence": confidence,
        "defective_probability": defective_probability,
        "ok_probability": ok_probability,
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            'python src\\models\\process_image.py '
            '"path\\to\\image.jpeg"'
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print("Image not found:", image_path)
        sys.exit(1)

    result = predict_image(image_path)

    print("=== Casting Image Prediction ===")
    print()
    print("Image:", image_path.name)
    print("Prediction:", result["prediction"])
    print(f"Confidence: {result['confidence']:.2%}")
    print(
        f"Defective probability: "
        f"{result['defective_probability']:.2%}"
    )
    print(
        f"OK probability: "
        f"{result['ok_probability']:.2%}"
    )