# LSD-TFFormer: Retinex-Based Transformer for Low-Light Image Enhancement

This repository contains the implementation of **LSD-TFFormer**, a Retinex-inspired Transformer model for **Low-Light Image Enhancement (LLIE)**.

The model decomposes an image into illumination and reflectance, enhances reflectance using window-based self-attention, applies denoising, and reconstructs a clean, bright image.

---

## 🚀 Features

- Retinex-based image decomposition
- Window-based Transformer attention (efficient)
- Reflectance enhancement module
- CNN-based denoiser
- Mixed precision (AMP) training
- Gradient clipping for stability
- Resume training from checkpoint
- PSNR & SSIM evaluation
- Separate train / validation / test scripts
- GPU support

---

## 📂 Project Structure

```
LSD-TFFormer-LLIE/
│
├── train.py
├── validate.py
├── test.py
├── config.py
├── requirements.txt
│
├── datasets/
│   └── llie_dataset.py
│
├── models/
│   ├── blocks.py
│   └── lsd_tf_former.py
│
├── utils/
│   ├── metrics.py
│   └── checkpoint.py
│
├── checkpoints/
├── results/
└── README.md
```

---

## 🧠 Model Architecture

The model consists of three main components:

### 1️⃣ Illumination Estimator
- Predicts a 1-channel illumination map.
- Uses CNN layers + Sigmoid activation.
- Prevents division instability using clamping.

### 2️⃣ Reflectance Restoration (Transformer)
- Extracts features using convolution.
- Uses multiple Transformer blocks.
- Applies window-based attention (8×8).
- Restores enhanced reflectance.

### 3️⃣ Denoiser
- CNN residual denoiser.
- Removes noise after enhancement.

### 🔁 Final Reconstruction

```
Reflectance = Input / Illumination
Enhanced = Transformer(Reflectance)
Denoised = Denoiser(Enhanced)
Output = Denoised × Illumination
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/LSD-TFFormer-LLIE.git
cd LSD-TFFormer-LLIE
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🖥️ GPU Check

Make sure CUDA is available:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## 📊 Dataset Structure

```
data/
│
├── train/
│   ├── low/
│   └── high/
│
├── val/
│   ├── low/
│   └── high/
│
└── test/
    └── low/
```

- `low` = dark images  
- `high` = ground-truth bright images  

Update dataset paths inside `config.py`.

---

## 🏋️ Training

To start training:

```bash
python train.py
```

Features:
- Automatic checkpoint saving
- Resume training if checkpoint exists
- Mixed precision (AMP)
- Gradient clipping

---

## 📈 Validation

To evaluate on validation set:

```bash
python validate.py
```

Outputs:
- PSNR (dB)
- SSIM

---

## 🧪 Testing (Inference Only)

To enhance test images:

```bash
python test.py
```

Enhanced images will be saved in:

```
results/
```

---

## ⚙️ Training Configuration

Edit `config.py`:

```python
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 150
IMG_SIZE = 256
```

---

## 📊 Evaluation Metrics

### PSNR
Peak Signal-to-Noise Ratio

### SSIM
Structural Similarity Index

Both are implemented manually in `utils/metrics.py`.

---

## 💾 Checkpointing

The model automatically saves:

- Model weights
- Optimizer state
- AMP scaler state
- Current epoch

Checkpoint file:
```
checkpoints/lsd_tf_checkpoint.pth
```

Training automatically resumes if checkpoint exists.

---

## 🔥 Performance Notes

Current setup:
- Loss: L1
- Optimizer: Adam
- Window size: 8
- Transformer blocks: 4

Performance can be improved by:
- Adding perceptual loss (VGG/LPIPS)
- Adding multi-scale training
- Adding illumination smoothness loss
- Using cosine LR scheduler

---

## 🛠 Requirements

- Python 3.8+
- PyTorch 2.x
- torchvision
- Pillow
- CUDA (recommended)

---

## 📌 Future Improvements

- Multi-scale Transformer
- Noise-aware enhancement
- Perceptual loss integration
- NTIRE competition optimization
- Real-time inference optimization

---

## 📜 License

This project is open-source and free to use for research and educational purposes.

---

## ⭐ If This Helps You

Please consider starring the repository.
