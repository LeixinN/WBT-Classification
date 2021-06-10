# Neural Network Files
# Author: Leixin NIE


# Import packages section
import torch
import torch.nn as nn


# Implement Logistic Regression via a single-layer neural network
class SoftMaxNet(nn.Module):
    def __init__(self, num_feature=4, num_classes=2):
        super(SoftMaxNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_feature, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define a convolutional neural network -- ConvNet2()
class ConvNet2(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, (2, 4)),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 5)),
            nn.Conv2d(12, 24, (2, 4)),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.matchlayer = nn.MaxPool2d((4, 5))
        self.classifier = nn.Sequential(
            nn.Linear(24*2*3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.matchlayer(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


