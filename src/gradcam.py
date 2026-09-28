from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from models.resnet18_model import CastingResNet18


MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "gradcam"
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# Same preprocessing used during ResNet18 training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():

    model = CastingResNet18().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    return model


def generate_gradcam(image_path):

    image_path = Path(image_path)

    model = load_model()

    # -----------------------------
    # Prepare image
    # -----------------------------

    original_image = Image.open(
        image_path
    ).convert("L")

    original_image = original_image.resize(
        (224, 224)
    )

    rgb_image = np.array(
        original_image.convert("RGB")
    ) / 255.0

    input_image = transform(
        Image.open(image_path)
    ).unsqueeze(0).to(device)


    # -----------------------------
    # Prediction
    # -----------------------------

    with torch.no_grad():

        output = model(input_image)

        defective_probability = torch.sigmoid(
            output
        ).item()

    ok_probability = 1 - defective_probability

    if defective_probability >= 0.5:

        prediction = "Defective"

        confidence = defective_probability

    else:

        prediction = "OK"

        confidence = ok_probability


    # -----------------------------
    # Grad-CAM
    # -----------------------------

    target_layers = [
        model.model.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # For our single-output sigmoid model,
    # use the model output itself as the target.
    targets = [
    ClassifierOutputTarget(0)
]
    grayscale_cam = cam(
        input_tensor=input_image,
        targets=targets
    )[0]


    # -----------------------------
    # Create visualization
    # -----------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # -----------------------------
    # Save result
    # -----------------------------

    output_path = (
        RESULTS_PATH
        / f"{image_path.stem}_gradcam.png"
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(visualization)

    plt.title(
        f"Predicted: {prediction} | "
        f"Confidence: {confidence:.2%}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    return {
        "prediction": prediction,
        "confidence": confidence,
        "defective_probability": defective_probability,
        "ok_probability": ok_probability,
        "gradcam_path": output_path,
    }


if __name__ == "__main__":

    sample_image = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "casting_data"
        / "casting_data"
        / "test"
        / "def_front"
        / "cast_def_0_1059.jpeg"
    )

    result = generate_gradcam(
        sample_image
    )

    print("=== GRAD-CAM EXPLANATION ===")
    print()

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.2%}"
    )

    print(
        f"Defective probability: "
        f"{result['defective_probability']:.2%}"
    )

    print(
        f"OK probability: "
        f"{result['ok_probability']:.2%}"
    )

    print(
        "Grad-CAM saved to:",
        result["gradcam_path"]
    )