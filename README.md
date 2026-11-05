# Multimodal Emotion Recognition 🎭

Emotions don't live in a single channel. When someone speaks, their *voice* carries tone and energy, while their *face* carries the expressions that go with it. This project tries to capture both — it's a deep learning model that watches a short video clip and listens to its audio, then decides which of **7 emotions** the person is feeling: Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral.

## How it works

The idea is simple: two separate streams process the two modalities, and a cross-attention layer lets the visual stream "look at" what the audio stream learned before making a final call.

- **Audio stream** — The raw audio is converted into a log-mel spectrogram (essentially a picture of sound), and a ResNet-18 treats it as an image to extract spectral features.
- **Visual stream** — 16 frames are sampled evenly from the video. A ResNet-18 extracts spatial features from each frame, and a transformer encoder layer models how expressions change over time.
- **Cross-attention fusion** — The visual embedding acts as the query and the audio embedding as key/value, so the model learns to align what it sees with what it hears. The fused representation then goes through a small classifier head.
- **Balanced loss** — On top of standard cross-entropy, there's an auxiliary contrastive loss (cosine similarity with a temperature of 0.07) that pulls matched audio-visual pairs together and pushes mismatched pairs apart. This keeps both streams meaningful instead of letting one dominate the other.

```
Audio (3s, 16kHz)                      Video (16 frames, 224x224)
      │                                        │
      ▼                                        ▼
Log-Mel Spectrogram (128 mels)         ResNet-18 (per frame)
      │                                        │
      ▼                                        ▼
ResNet-18 (1-channel)             Transformer Encoder (temporal)
      │                                        │
      ▼                                        ▼
 Audio Embedding (256) ◄── Cross-Attention ──► Visual Embedding (256)
      │              (visual = query,               │
      │               audio = key/value)            │
      └────────────────┬───────────────────────────┘
                       ▼
              Fused Representation
                       ▼
           Classifier → 7 Emotions
```

## Project structure

```
├── config.py                # All the knobs: sample rate, frame count, embed dim, etc.
├── dataset.py               # Dataset that loads audio → spectrogram and video → frames
├── loss.py                  # Cross-entropy + contrastive alignment loss
├── train.py                 # Training loop (one epoch + main entry point)
└── models/
    ├── audio_stream.py      # Audio encoder (ResNet-18 on spectrograms)
    ├── visual_stream.py     # Visual encoder (ResNet-18 + temporal transformer)
    └── fusion.py            # Cross-attention fusion + classifier
```

## Getting started

Clone the repo and install the dependencies:

```bash
git clone https://github.com/Pratham1-hue/Multimodal-Emotion.git
cd Multimodal-Emotion
pip install torch torchvision torchaudio librosa opencv-python numpy
```

Training expects the dataset as a list of samples, where each sample points to an audio file, a video file, and an emotion label:

```python
samples = [
    {"audio_path": "clips/1.wav", "video_path": "clips/1.mp4", "label": 3},
]
```

Then run:

```bash
python train.py
```

The model will automatically use your GPU if CUDA is available, otherwise it falls back to CPU.

## Key hyperparameters

| Parameter | Value |
|---|---|
| Sample rate | 16 kHz |
| Audio length | 3 seconds |
| Mel bins | 128 |
| Video frames | 16 @ 224×224 |
| Embedding dim | 256 |
| Classes | 7 |
| Batch size | 16 |
| Learning rate | 1e-4 (AdamW) |
| Epochs | 20 |
| Aux (contrastive) weight | 0.2 |

All of these live in `config.py`, so tweaking experiments is just editing one file.

## Screenshots

Training curves and sample predictions will go here as the project progresses:

| Training Loss / Accuracy | Architecture |

<p align="center">
  <img src="docs/screenshot_loss.png" alt="Training loss and accuracy curves" width="45%">
  <img src="docs/screenshot_diagram.png" alt="Model architecture diagram" width="45%">
</p>

*(Placeholders — drop your training output images into a `docs/` folder to replace these.)*

## Ideas for next steps

- [ ] Add a validation loop and metrics like per-class F1 (emotions are imbalanced!)
- [ ] Checkpointing and early stopping
- [ ] Data augmentations (SpecAugment for audio, random crops for frames)
- [ ] An inference script to run the model on a single video
- [ ] Evaluate on standard benchmarks like RAVDESS or CREMA-D

## License

MIT — do whatever you like with it.

---

Built with PyTorch. If you find this useful, feel free to ⭐ the repo!
