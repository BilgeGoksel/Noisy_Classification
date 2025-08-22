import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DeepCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 128 x 1 x 1
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)

# ResNet34 (pretrained & fine-tuned)
class ResNet34Pretrained(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m
    def forward(self, x): return self.model(x)



# ResNet34 (scratch)
class ResNet34Scratch(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m
    def forward(self, x): return self.model(x)


# EfficientNetB0
class EfficientNetB0Custom(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = m
    def forward(self, x): return self.model(x)