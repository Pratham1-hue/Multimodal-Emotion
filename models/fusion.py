import torch
import torch.nn as nn
from .audio_stream import AudioStream
from .visual_stream import VisualStream

class CrossAttentionFusion(nn.Module):
    """Dual-stream fusion aligning visual frame queries with audio spectral keys."""
    def __init__(self, embed_dim=256, num_classes=7):
        super().__init__()
        self.audio_net = AudioStream(embed_dim)
        self.visual_net = VisualStream(embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio_x, visual_x):
        audio_emb = self.audio_net(audio_x)     # (B, embed_dim)
        visual_emb = self.visual_net(visual_x)   # (B, embed_dim)
        
        q = visual_emb.unsqueeze(1)
        k = v = audio_emb.unsqueeze(1)
        
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        attn_out = self.norm(q + attn_out).squeeze(1)
        
        fused = torch.cat([attn_out, audio_emb], dim=-1)
        return self.classifier(fused), audio_emb, visual_emb