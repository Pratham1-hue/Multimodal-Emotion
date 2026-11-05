import torch
import torch.nn as nn
import torchvision.models as models

class AudioStream(nn.Module):
    """Extracts 2D spectral features from audio spectrograms."""
    def __init__(self, embed_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Adapt first conv layer from 3 RGB channels to 1 audio channel
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        # x shape: (B, 1, N_MELS, Time)
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        return self.fc(feat)