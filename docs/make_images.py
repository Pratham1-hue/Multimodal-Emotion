"""Generate README images: architecture diagram and sample training curves."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------- Architecture diagram ----------------
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=140)
ax.set_xlim(0, 11)
ax.set_ylim(0, 10)
ax.axis("off")

AUDIO = "#2563eb"   # blue
VIDEO = "#16a34a"   # green
FUSED = "#7c3aed"   # purple
GRAY  = "#334155"

def box(x, y, w, h, text, color, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=color, ec="none", alpha=0.92))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color="white", fontsize=fs, weight="bold")

def arrow(x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=1.8))

# Input labels
ax.text(2.75, 9.65, "Audio input\n(3 s, 16 kHz)", ha="center", color=AUDIO, fontsize=11, weight="bold")
ax.text(8.25, 9.65, "Video input\n(16 frames, 224×224)", ha="center", color=VIDEO, fontsize=11, weight="bold")

# Audio column
box(1.25, 8.1, 3.0, 1.0, "Log-Mel Spectrogram\n(128 mels)", AUDIO)
box(1.25, 6.4, 3.0, 1.0, "ResNet-18\n(1-channel backbone)", AUDIO)
box(1.25, 4.7, 3.0, 1.0, "Audio embedding\n(256-d)", AUDIO)

# Video column
box(6.75, 8.1, 3.0, 1.0, "Frame sampling\n(16 frames)", VIDEO)
box(6.75, 6.4, 3.0, 1.0, "ResNet-18 (per frame)\n+ Transformer encoder", VIDEO)
box(6.75, 4.7, 3.0, 1.0, "Visual embedding\n(256-d)", VIDEO)

# Fusion
box(4.0, 3.0, 3.0, 1.0, "Cross-Attention Fusion\n(query=visual, kv=audio)", FUSED, fs=9.5)
box(4.0, 1.6, 3.0, 0.85, "Classifier MLP", GRAY, fs=10)
box(4.0, 0.3, 3.0, 0.85, "7 Emotions\n😊 😠 😨 …", "#dc2626", fs=10)

# Arrows
arrow(2.75, 9.3, 2.75, 8.35, AUDIO)
arrow(2.75, 8.0, 2.75, 7.5, AUDIO)
arrow(2.75, 6.3, 2.75, 5.8, AUDIO)
arrow(8.25, 9.3, 8.25, 8.35, VIDEO)
arrow(8.25, 8.0, 8.25, 7.5, VIDEO)
arrow(8.25, 6.3, 8.25, 5.8, VIDEO)
arrow(2.75, 4.6, 4.6, 4.1, AUDIO)
arrow(8.25, 4.6, 6.4, 4.1, VIDEO)
arrow(5.5, 2.9, 5.5, 2.5)
arrow(5.5, 1.5, 5.5, 1.2)

# Contrastive loss annotation
ax.text(0.15, 3.35, "Contrastive alignment loss\n(cosine similarity, τ = 0.07)",
        ha="left", color=FUSED, fontsize=9.5, style="italic")

fig.savefig("docs/architecture.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- Sample training curves ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)

epochs = list(range(1, 21))
train_loss = [1.85, 1.62, 1.44, 1.30, 1.18, 1.08, 1.00, 0.93, 0.86, 0.81,
              0.76, 0.72, 0.68, 0.64, 0.61, 0.58, 0.55, 0.52, 0.50, 0.48]
val_loss   = [1.80, 1.63, 1.52, 1.43, 1.36, 1.30, 1.26, 1.22, 1.20, 1.18,
              1.17, 1.16, 1.16, 1.15, 1.16, 1.17, 1.18, 1.19, 1.21, 1.22]
acc_train = [0.36, 0.44, 0.50, 0.55, 0.59, 0.63, 0.66, 0.68, 0.71, 0.73,
             0.75, 0.76, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85]
acc_val =   [0.35, 0.42, 0.47, 0.51, 0.55, 0.58, 0.60, 0.62, 0.64, 0.65,
             0.66, 0.66, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67]

ax = axes[0]
ax.plot(epochs, train_loss, "-o", ms=3.5, color="#2563eb", label="train loss")
ax.plot(epochs, val_loss, "-s", ms=3.5, color="#dc2626", label="val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(epochs, acc_train, "-o", ms=3.5, color="#16a34a", label="train acc")
ax.plot(epochs, acc_val, "-s", ms=3.5, color="#ea580c", label="val acc")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Accuracy"); ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Sample training curves (synthetic example — replace with real run)", fontsize=11, y=1.02)
fig.savefig("docs/training_curves.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("done")
