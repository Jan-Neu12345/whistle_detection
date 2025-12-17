import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F


def get_model(device): # -> models.resnet.ResNet:
    #model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 2)
    
    #model = models.mobilenet_v3_small(num_classes=2)
    model = models.efficientnet_b0(weights="DEFAULT")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model.to(device)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    