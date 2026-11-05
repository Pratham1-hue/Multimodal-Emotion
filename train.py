import torch
from torch.utils.data import DataLoader
import config
from models.fusion import CrossAttentionFusion
from loss import BalancedMultimodalLoss

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for audio, video, targets in loader:
        audio = audio.to(config.DEVICE)
        video = video.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        
        optimizer.zero_grad()
        logits, audio_emb, visual_emb = model(audio, video)
        loss = criterion(logits, audio_emb, visual_emb, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    return total_loss / total, correct / total

if __name__ == "__main__":
    model = CrossAttentionFusion(embed_dim=config.EMBED_DIM, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = BalancedMultimodalLoss(aux_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
    
    print(f"Model initialized on device: {config.DEVICE}")