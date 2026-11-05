import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedMultimodalLoss(nn.Module):
    """Cross-Entropy + Contrastive alignment penalty to balance audio and video streams."""
    def __init__(self, aux_weight=0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.aux_weight = aux_weight

    def forward(self, main_logits, audio_emb, visual_emb, targets):
        main_loss = self.ce(main_logits, targets)
        
        # Normalize vectors for cosine similarity alignment
        audio_norm = F.normalize(audio_emb, dim=-1)
        visual_norm = F.normalize(visual_emb, dim=-1)
        
        sim_matrix = torch.matmul(audio_norm, visual_norm.T) / 0.07
        labels = torch.arange(audio_emb.size(0), device=audio_emb.device)
        aux_loss = self.ce(sim_matrix, labels)
        
        return main_loss + (self.aux_weight * aux_loss)