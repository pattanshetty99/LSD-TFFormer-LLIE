import torch

# =====================
# PATHS (EDIT THESE)
# =====================
TRAIN_LOW = "data/train/low"
TRAIN_HIGH = "data/train/high"

VAL_LOW = "data/val/low"
VAL_HIGH = "data/val/high"

CHECKPOINT_PATH = "checkpoints/lsd_tf_checkpoint.pth"

# =====================
# TRAIN SETTINGS
# =====================
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 150
IMG_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
