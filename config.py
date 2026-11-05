import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio parameters
SAMPLE_RATE = 16000
N_MELS = 128
AUDIO_LEN_SEC = 3.0

# Video parameters
NUM_FRAMES = 16
FRAME_SIZE = (224, 224)

# Training hyperparameters
EMBED_DIM = 256
NUM_CLASSES = 7  # Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20