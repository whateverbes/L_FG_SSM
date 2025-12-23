import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# --- 1. 轴向轻量化卷积 (保持 LU2Net 的实时性) ---
class AxialDWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# --- 2. 交互增强模块 (核心优化：频域+Mamba) ---
class FSIBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 空间支路: Mamba
        if Mamba:
            self.spatial_mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=1)
        else:
            self.spatial_mamba = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1), nn.Sigmoid())

        # 频率支路: 幅度调制
        self.amp_modulator = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim), nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        amp, phase = torch.abs(x_fft), torch.angle(x_fft)
        weight = self.amp_modulator(torch.mean(x, dim=(2, 3))).view(B, C, 1, 1)
        x_freq = torch.fft.irfft2(amp * weight * torch.exp(1j * phase), s=(H, W), norm='ortho')

        if Mamba:
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            x_spat = self.spatial_mamba(x_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x_spat = x * self.spatial_mamba(x)
        return x_freq + x_spat


# --- 3. 最终优化架构 ---
class L_FG_SSM(nn.Module):
    def __init__(self, base_dim=32, num_classes=5):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, base_dim, 3, 1, 1), AxialDWConv(base_dim))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), AxialDWConv(base_dim))

        # Bottleneck
        self.bottleneck = FSIBlock(base_dim)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv = AxialDWConv(base_dim)

        # 分支 A: 增强输出 (3通道图像)
        self.enhance_head = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)

        # 分支 B: 分割输出 (5通道掩码，对应 LIACI 4类缺陷+背景)
        self.task_head = nn.Sequential(
            nn.Conv2d(base_dim, base_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d = self.up(b)
        # 融合浅层和深层特征
        d = self.dec_conv(d + e1)

        # 同时输出图像和分割结果
        out_img = self.enhance_head(d)
        out_mask = self.task_head(d)
        return out_img, out_mask
