import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from config import *
from datasets.llie_dataset import LLIE_Dataset
from models.lsd_tf_former import LSD_TFFormer
from utils.checkpoint import save_checkpoint, load_checkpoint

train_dataset = LLIE_Dataset(TRAIN_LOW, TRAIN_HIGH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = LSD_TFFormer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()
scaler = GradScaler('cuda')

start_epoch = load_checkpoint(model, optimizer, scaler, CHECKPOINT_PATH, DEVICE)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    for low, high in train_loader:
        low, high = low.to(DEVICE), high.to(DEVICE)

        optimizer.zero_grad()

        with autocast('cuda'):
            output, _ = model(low)
            loss = criterion(output, high)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    save_checkpoint(model, optimizer, scaler, epoch+1, CHECKPOINT_PATH)
