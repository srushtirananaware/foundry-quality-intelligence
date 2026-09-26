from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.resnet18_model import CastingResNet18


# -----------------------------------
# Project paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "casting_data"
    / "casting_data"
    / "test"
)

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"

RESULTS_PATH = PROJECT_ROOT / "results" / "gradcam"

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------------
# Image preprocessing
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------------
# Load model
# -----------------------------------

model = CastingResNet18().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# -----------------------------------
# Grad-CAM target layer
# -----------------------------------

target_layers = [
    model.model.layer4[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# -----------------------------------
# Process one image
# -----------------------------------

def generate_gradcam(
    image_path,
    output_path,
    actual_class
):

    print("\nProcessing:", image_path.name)

    # Load image
    original_image = Image.open(image_path).convert("L")

    original_image = original_image.resize(
        (224, 224)
    )

    # Image for visualization
    rgb_image = np.array(
        original_image.convert("RGB")
    ) / 255.0

    # Image for model
    input_image = transform(
        Image.open(image_path)
    ).unsqueeze(0).to(device)


    # -----------------------------------
    # Prediction
    # -----------------------------------

    with torch.no_grad():

        output = model(input_image)

        ok_probability = torch.sigmoid(
            output
        ).item()


    defective_probability = 1 - ok_probability

    prediction = (
        "OK"
        if ok_probability >= 0.5
        else "Defective"
    )


    print("Actual class:", actual_class)
    print("Prediction:", prediction)

    print(
        "Defective probability:",
        round(defective_probability, 4)
    )

    print(
        "OK probability:",
        round(ok_probability, 4)
    )


    # -----------------------------------
    # Grad-CAM
    # -----------------------------------

    targets = [
        ClassifierOutputTarget(0)
    ]

    grayscale_cam = cam(
        input_tensor=input_image,
        targets=targets
    )

    grayscale_cam = grayscale_cam[0]


    # -----------------------------------
    # Overlay heatmap
    # -----------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # -----------------------------------
    # Save visualization
    # -----------------------------------

    plt.figure(figsize=(8, 8))

    plt.imshow(visualization)

    plt.title(
        f"Actual: {actual_class} | "
        f"Predicted: {prediction}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# -----------------------------------
# Find example images
# -----------------------------------

defective_images = list(
    (DATASET_PATH / "def_front").glob("*.jpeg")
)

if not defective_images:
    defective_images = list(
        (DATASET_PATH / "def_front").glob("*.jpg")
    )


ok_images = list(
    (DATASET_PATH / "ok_front").glob("*.jpeg")
)

if not ok_images:
    ok_images = list(
        (DATASET_PATH / "ok_front").glob("*.jpg")
    )


# -----------------------------------
# Generate both examples
# -----------------------------------

generate_gradcam(
    defective_images[0],
    RESULTS_PATH / "defective_example.png",
    "Defective"
)


generate_gradcam(
    ok_images[0],
    RESULTS_PATH / "ok_example.png",
    "OK"
)


print("\n" + "=" * 50)
print("GRAD-CAM RESULTS COMPLETE")
print("=" * 50)