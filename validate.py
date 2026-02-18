import torch
from torch.utils.data import DataLoader

from config import *
from datasets.llie_dataset import LLIE_Dataset
from models.lsd_tf_former import LSD_TFFormer
from utils.metrics import calculate_psnr, calculate_ssim

val_dataset = LLIE_Dataset(VAL_LOW, VAL_HIGH, IMG_SIZE)
val_loader = DataLoader(val_dataset, batch_size=1)

model = LSD_TFFormer().to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

total_psnr = 0
total_ssim = 0

with torch.no_grad():
    for low, high in val_loader:
        low, high = low.to(DEVICE), high.to(DEVICE)
        output, _ = model(low)

        total_psnr += calculate_psnr(output, high).item()
        total_ssim += calculate_ssim(output, high).item()

print("\nValidation Results")
print(f"PSNR : {total_psnr/len(val_loader):.2f} dB")
print(f"SSIM : {total_ssim/len(val_loader):.4f}")
