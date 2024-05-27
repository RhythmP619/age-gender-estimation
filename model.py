import torch
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.2),
            
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.2),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.features(x)

class AgeEstimator(nn.Module):
    def __init__(self):
        super(AgeEstimator, self).__init__()
        self.base_cnn = BaseCNN()
        self.classifier = nn.Sequential(
            nn.Linear(256*2*2, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.base_cnn(x)
        x = self.classifier(x)
        return x

class GenderEstimator(nn.Module):
    def __init__(self):
        super(GenderEstimator, self).__init__()
        self.base_cnn = BaseCNN()
        self.classifier = nn.Sequential(
            nn.Linear(256*2*2, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),

            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.base_cnn(x)
        x = self.classifier(x)
        return x