import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LLIE_Dataset(Dataset):
    def __init__(self, low_dir, high_dir, img_size=256):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.files = sorted([
            f for f in os.listdir(low_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        low = Image.open(os.path.join(self.low_dir, filename)).convert("RGB")
        high = Image.open(os.path.join(self.high_dir, filename)).convert("RGB")

        return self.transform(low), self.transform(high)

