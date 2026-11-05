import torch
import torch.nn as nn
import torchvision.models as models

class VisualStream(nn.Module):
    """Extracts spatial-temporal features from sampled video frame sequences."""
    def __init__(self, embed_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.spatial = nn.Sequential(*list(resnet.children())[:-1])
        
        self.temporal = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=1024, batch_first=True
        )
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        # x shape: (B, NUM_FRAMES, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        spatial_feats = self.spatial(x).view(B, T, 512)
        temporal_feats = self.temporal(spatial_feats)
        
        pooled = temporal_feats.mean(dim=1)
        return self.fc(pooled)