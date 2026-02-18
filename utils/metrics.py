import torch
import torch.nn.functional as F

def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11):
    mu1 = F.avg_pool2d(img1, window_size, 1, 0)
    mu2 = F.avg_pool2d(img2, window_size, 1, 0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, 0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, 0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, 0) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()
