import torch.nn as nn
from torchvision import models


def get_model(device): # -> models.resnet.ResNet:
    model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    #model = models.mobilenet_v3_small(num_classes=2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model.to(device)
