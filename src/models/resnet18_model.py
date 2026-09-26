import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CastingResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        # Load ResNet18 with pretrained ImageNet weights
        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Number of inputs to the original final layer
        num_features = self.model.fc.in_features

        # Replace ImageNet's 1000-class classifier
        # with our binary casting classifier
        self.model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    import torch

    model = CastingResNet18()

    dummy_image = torch.randn(1, 3, 224, 224)

    output = model(dummy_image)

    print(model)
    print("\nInput shape:", dummy_image.shape)
    print("Output shape:", output.shape)