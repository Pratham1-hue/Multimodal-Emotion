import torch
import librosa
import cv2
import numpy as np
from torch.utils.data import Dataset
import config

class MultimodalEmotionDataset(Dataset):
    def __init__(self, samples):
        """samples: list of dicts [{'audio_path': ..., 'video_path': ..., 'label': int}]"""
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        y, sr = librosa.load(path, sr=config.SAMPLE_RATE, duration=config.AUDIO_LEN_SEC)
        target_len = int(config.SAMPLE_RATE * config.AUDIO_LEN_SEC)
        
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
            
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        return torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(0, total_frames - 1), config.NUM_FRAMES, dtype=int)
        
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                frame = cv2.resize(frame, config.FRAME_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                frames.append(frame)
            idx += 1
        cap.release()
        
        while len(frames) < config.NUM_FRAMES:
            frames.append(np.zeros((*config.FRAME_SIZE, 3)))
            
        return torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio = self._load_audio(sample['audio_path'])
        video = self._load_video(sample['video_path'])
        label = torch.tensor(sample['label'], dtype=torch.long)
        return audio, video, label