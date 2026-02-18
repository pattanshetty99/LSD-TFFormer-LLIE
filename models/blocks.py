import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, heads=4):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws

        if H % ws != 0 or W % ws != 0:
            pad_h = ws - H % ws
            pad_w = ws - W % ws
            x = F.pad(x, (0, pad_w, 0, pad_h))
            B, C, H, W = x.shape

        x = x.view(B, C, H//ws, ws, W//ws, ws)
        x = x.permute(0,2,4,3,5,1).contiguous()
        x = x.view(-1, ws*ws, C)

        qkv = self.qkv(x)
        qkv = qkv.reshape(-1, ws*ws, 3, self.heads, C//self.heads)
        qkv = qkv.permute(2,0,3,1,4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1,2).reshape(-1, ws*ws, C)
        out = self.proj(out)

        out = out.view(B, H//ws, W//ws, ws, ws, C)
        out = out.permute(0,5,1,3,2,4).contiguous()
        out = out.view(B, C, H, W)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = WindowAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
