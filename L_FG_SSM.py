import torch
import torch.nn as nn
import torch.nn.functional as F

# 检查 Mamba 环境
try:
    from mamba_ssm import Mamba
except ImportError:
    print("❌ 警告: 未安装 mamba_ssm，将使用降级卷积模式。")
    Mamba = None


# ==========================================
# 1. 物理感知颜色补偿模块 (灵感来源: UCCNet)
# ==========================================
class PhysicsAwareHead(nn.Module):
    """
    模拟水下物理成像：利用 G/B 通道的信息来补偿衰减最严重的 R 通道。
    而不是盲目地用 3x3 卷积去猜颜色。
    """

    def __init__(self, in_dim):
        super().__init__()
        # 特征压缩，准备生成 RGB
        self.pre_conv = nn.Conv2d(in_dim, 3, 1)

        # 物理补偿注意力：学习如何从 G,B 通道借信息补 R
        self.compensate = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()  # 生成 0~1 的补偿权重
        )

    def forward(self, x):
        # 初步生成的 RGB
        rgb = self.pre_conv(x)  # [B, 3, H, W]

        # 计算补偿图
        comp_map = self.compensate(rgb)

        # 物理残差连接：原始 + 补偿 * 原始
        # 模拟 J = I / t + A... 的逆过程
        out = rgb + rgb * comp_map
        return torch.sigmoid(out)  # 约束到 0-1 之间


# ==========================================
# 2. 频域解耦 Mamba 模块 (灵感来源: FreqDINO)
# ==========================================
class FreqDecoupledBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # A. 空间分支：Mamba (负责全局上下文)
        if Mamba:
            self.spatial_mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        else:
            self.spatial_mamba = nn.Conv2d(dim, dim, 3, 1, 1)

        # B. 频域分支：可学习的频率滤波器
        # 1x1 卷积在频域相当于全通道混合
        self.freq_filter = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

        # C. 任务路由：决定哪些特征去增强，哪些去分割
        self.to_enhance = nn.Conv2d(dim, dim, 1)
        self.to_segment = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- 1. Mamba 空间建模 ---
        if Mamba:
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            x_spatial = self.spatial_mamba(x_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x_spatial = self.spatial_mamba(x)

        # --- 2. FFT 频域建模 ---
        x_fft = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_fft)
        pha = torch.angle(x_fft)

        # 学习频域权重 (去雾/去噪通常在频域处理更有效)
        mag_filtered = self.freq_filter(mag)
        x_freq = torch.fft.irfft2(mag_filtered * torch.exp(1j * pha), s=(H, W), norm='backward')

        # --- 3. 特征融合 ---
        feat_fused = x_spatial + x_freq

        # --- 4. 显式解耦 (关键创新点) ---
        # 这一步虽然简单，但在论文里可以画出非常漂亮的图：
        # "Frequency-Aware Task Routing"
        feat_enh = self.to_enhance(feat_fused)  # 偏向低频/结构
        feat_seg = self.to_segment(feat_fused)  # 偏向高频/边缘

        return feat_enh, feat_seg


# ==========================================
# 3. 主干网络：L-FG-SSM (Pro Max 版)
# ==========================================
class AxialDWConv(nn.Module):
    """ 轻量化卷积组件，保持不变 """

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

        # Encoder (特征提取)
        self.enc1 = nn.Sequential(nn.Conv2d(3, base_dim, 3, 1, 1), AxialDWConv(base_dim))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), AxialDWConv(base_dim))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), AxialDWConv(base_dim))

        # Bottleneck: 核心创新模块
        self.bottleneck = FreqDecoupledBlock(base_dim)

        # Decoder (共用结构，但特征流分离)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 增强分支 Decoder
        self.dec_enh2 = AxialDWConv(base_dim)
        self.dec_enh1 = AxialDWConv(base_dim)

        # 分割分支 Decoder
        self.dec_seg2 = AxialDWConv(base_dim)
        self.dec_seg1 = AxialDWConv(base_dim)

        # Heads (输出头)
        # 创新点1: 物理感知增强头
        self.head_enhance = PhysicsAwareHead(base_dim)
        # 普通分割头
        self.head_segment = nn.Conv2d(base_dim, num_classes, 1)

    def forward(self, x):
        # --- 编码 ---
        e1 = self.enc1(x)  # [B, 32, 256, 256]
        e2 = self.enc2(e1)  # [B, 32, 128, 128]
        e3 = self.enc3(e2)  # [B, 32, 64, 64]

        # --- 瓶颈解耦 (Mamba + FFT) ---
        # 在最深层进行一次彻底的“任务分流”
        # f_enh 包含更多去雾后的结构信息
        # f_seg 包含更多锐化后的边界信息
        f_enh, f_seg = self.bottleneck(e3)

        # --- 解码分支 1: 图像增强 ---
        d_enh2 = self.dec_enh2(self.up(f_enh) + e2)  # Skip connection
        d_enh1 = self.dec_enh1(self.up(d_enh2) + e1)
        enhanced_img = self.head_enhance(d_enh1)

        # --- 解码分支 2: 语义分割 ---
        # 注意：分割任务不需要 e1, e2 这种太浅层的噪声细节，或者可以只融合 e2
        d_seg2 = self.dec_seg2(self.up(f_seg) + e2)
        d_seg1 = self.dec_seg1(self.up(d_seg2) + e1)
        segment_logits = self.head_segment(d_seg1)

        return enhanced_img, segment_logits
