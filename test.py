import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from config import *
from models.lsd_tf_former import LSD_TFFormer
from utils.metrics import calculate_psnr, calculate_ssim

# ============================
# SETTINGS
# ============================

TEST_INPUT_DIR = "data/test/low"
TEST_GT_DIR = None  # Set to "data/test/high" if GT available
SAVE_DIR = "results"

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================
# LOAD MODEL
# ============================

device = DEVICE

model = LSD_TFFormer().to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

print("✅ Model loaded successfully")

# ============================
# TRANSFORM
# ============================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

# ============================
# TESTING LOOP
# ============================

total_psnr = 0
total_ssim = 0
count = 0

with torch.no_grad():

    for filename in tqdm(sorted(os.listdir(TEST_INPUT_DIR))):

        input_path = os.path.join(TEST_INPUT_DIR, filename)
        image = Image.open(input_path).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.amp.autocast("cuda"):
            output, _ = model(input_tensor)

        output = output.squeeze(0).cpu().clamp(0, 1)

        # Save image
        save_path = os.path.join(SAVE_DIR, filename)
        to_pil(output).save(save_path)

        # If GT available, compute metrics
        if TEST_GT_DIR is not None:
            gt_path = os.path.join(TEST_GT_DIR, filename)
            gt_image = Image.open(gt_path).convert("RGB")
            gt_tensor = transform(gt_image).unsqueeze(0).to(device)

            psnr = calculate_psnr(output.unsqueeze(0).to(device), gt_tensor)
            ssim = calculate_ssim(output.unsqueeze(0).to(device), gt_tensor)

            total_psnr += psnr.item()
            total_ssim += ssim.item()
            count += 1

# ============================
# PRINT RESULTS
# ============================

if TEST_GT_DIR is not None and count > 0:
    print("\nTest Results")
    print(f"PSNR : {total_psnr/count:.2f} dB")
    print(f"SSIM : {total_ssim/count:.4f}")

print(f"\n✅ Enhanced images saved in: {SAVE_DIR}")
