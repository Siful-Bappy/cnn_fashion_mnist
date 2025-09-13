import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, input_feature):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_feature, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 10)  # Assuming 10 classes for classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
