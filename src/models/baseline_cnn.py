import torch
import torch.nn as nn


class CastingCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Convolution Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolution Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolution Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 300x300 → 150x150 → 75x75 → 37x37
            nn.Linear(64 * 37 * 37, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            # Binary classification
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":

    model = CastingCNN()

    dummy_image = torch.randn(1, 1, 300, 300)

    output = model(dummy_image)

    print(model)
    print("\nInput shape:", dummy_image.shape)
    print("Output shape:", output.shape)
