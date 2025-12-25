import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 检查 Mamba 环境
try:
    from mamba_ssm import Mamba
except ImportError:
    print("❌ 警告: 未安装 mamba_ssm，将使用降级卷积模式。")
    Mamba = None

# ==========================================
# 1. 物理感知颜色补偿模块
# ==========================================
class PhysicsAwareHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_dim, 3, 1)
        
        # 引入全局池化来感知环境光 (Ambient Light)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.compensate = nn.Sequential(
            nn.Conv2d(3, 16, 1), # 处理全局特征
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        rgb = self.pre_conv(x) 
        
        # 基于全局颜色分布计算补偿系数
        global_feat = self.global_pool(rgb)
        comp_weight = self.compensate(global_feat) # [B, 3, 1, 1]
        
        # 物理残差: 原始图 + (原始图 * 补偿系数)
        out = rgb + rgb * comp_weight
        return torch.sigmoid(out)

# ==========================================
# 2. 频域解耦 Mamba 模块 (Pro Max Plus)
# ==========================================
class FreqDecoupledBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # A. 空间 Mamba
        if Mamba:
            self.spatial_mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        else:
            self.spatial_mamba = nn.Conv2d(dim, dim, 3, 1, 1)

        # B. 显式可学习频域掩码 (256x256 输入尺寸假设)
        # 初始化为中心为1(低频)，四周为0(高频)
        self.freq_mask = nn.Parameter(torch.randn(1, 1, 256, 129), requires_grad=True)
        
        # C. 任务路由
        self.to_enhance = nn.Conv2d(dim, dim, 1)
        self.to_segment = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Mamba 空间建模
        if Mamba:
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            x_spatial = self.spatial_mamba(x_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x_spatial = self.spatial_mamba(x)

        # 2. 频域解耦
        x_fft = torch.fft.rfft2(x, norm='backward') # [B, C, H, W/2+1]
        
        # 动态调整掩码尺寸以适应输入
        mask = torch.sigmoid(self.freq_mask)
        if mask.shape[2:] != x_fft.shape[2:]:
            mask = F.interpolate(mask, size=x_fft.shape[2:], mode='bilinear')
            
        # 低频 (结构/颜色) -> 增强
        x_low = x_fft * mask
        x_low_spatial = torch.fft.irfft2(x_low, s=(H, W), norm='backward')
        
        # 高频 (边缘/细节) -> 分割
        x_high = x_fft * (1 - mask)
        x_high_spatial = torch.fft.irfft2(x_high, s=(H, W), norm='backward')

        # 3. 融合与分流
        # 增强流 = 空间 + 低频
        feat_enh = self.to_enhance(x_spatial + x_low_spatial)
        # 分割流 = 空间 + 高频
        feat_seg = self.to_segment(x_spatial + x_high_spatial)
        
        return feat_enh, feat_seg

# ==========================================
# 3. 主干网络
# ==========================================
class AxialDWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class L_FG_SSM(nn.Module):
    def __init__(self, base_dim=32, num_classes=5):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, base_dim, 3, 1, 1), AxialDWConv(base_dim))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), AxialDWConv(base_dim))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), AxialDWConv(base_dim))

        self.bottleneck = FreqDecoupledBlock(base_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec_enh2 = AxialDWConv(base_dim)
        self.dec_enh1 = AxialDWConv(base_dim)
        self.dec_seg2 = AxialDWConv(base_dim)
        self.dec_seg1 = AxialDWConv(base_dim)

        self.head_enhance = PhysicsAwareHead(base_dim)
        self.head_segment = nn.Conv2d(base_dim, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        f_enh, f_seg = self.bottleneck(e3)
        
        # 增强流
        d_enh2 = self.dec_enh2(self.up(f_enh) + e2)
        d_enh1 = self.dec_enh1(self.up(d_enh2) + e1)
        enhanced_img = self.head_enhance(d_enh1)
        
        # 分割流 (深层特征更重要，浅层e1包含噪声，这里选择只融合e2或都不融合，这里保留融合)
        d_seg2 = self.dec_seg2(self.up(f_seg) + e2)
        d_seg1 = self.dec_seg1(self.up(d_seg2) + e1)
        segment_logits = self.head_segment(d_seg1)
        
        return enhanced_img, segment_logits
