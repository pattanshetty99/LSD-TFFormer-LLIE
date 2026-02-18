import torch
import os

def save_checkpoint(model, optimizer, scaler, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict()
    }, path)


def load_checkpoint(model, optimizer, scaler, path, device):
    if os.path.exists(path):
        print("Loading checkpoint...")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        return checkpoint["epoch"]
    return 0
