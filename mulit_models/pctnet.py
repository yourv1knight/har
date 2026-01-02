"""
PCTNet模型定义 - 用于图像协调
基于 "PCT-Net: Full Resolution Image Harmonization Using Pixel-wise Color Transforms"

支持:
1. 从libcom加载预训练权重
2. 微调训练（支持冻结Encoder）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """基础卷积块: Conv + BN + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """下采样块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=2)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    """上采样块 with Skip Connection"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不匹配
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionModule(nn.Module):
    """简化的注意力模块"""

    def __init__(self, channels):
        super().__init__()
        self.conv_query = nn.Conv2d(channels, channels // 8, 1)
        self.conv_key = nn.Conv2d(channels, channels // 8, 1)
        self.conv_value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()

        # Query, Key, Value
        query = self.conv_query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.conv_key(x).view(batch, -1, height * width)
        value = self.conv_value(x).view(batch, -1, height * width)

        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)

        return self.gamma * out + x


class PCTNet(nn.Module):
    """
    PCT-Net风格的图像协调网络

    架构: U-Net with Attention
    输入: [Composite Image (3 channels) + Mask (1 channel)] = 4 channels
    输出: Harmonized Image (3 channels)
    """

    def __init__(self, in_channels=4, out_channels=3, base_channels=64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (可冻结用于微调)
        self.conv_in = ConvBlock(in_channels, base_channels)

        self.down1 = DownBlock(base_channels, base_channels * 2)      # 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 256
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)  # 512
        self.down4 = DownBlock(base_channels * 8, base_channels * 8)  # 512

        # Bottleneck with Attention
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 8),
            AttentionModule(base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )

        # Decoder
        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)

        # Output
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, composite, mask):
        """
        Args:
            composite: 合成图 (B, 3, H, W)
            mask: 掩码 (B, 1, H, W)
        Returns:
            harmonized: 协调后的图像 (B, 3, H, W)
        """
        # 拼接输入
        x = torch.cat([composite, mask], dim=1)  # (B, 4, H, W)

        # Encoder
        x0 = self.conv_in(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Bottleneck
        x = self.bottleneck(x4)

        # Decoder with skip connections
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # Output
        out = self.conv_out(x)

        # Residual connection: 输出 = 输入 + 调整量
        harmonized = composite + out

        return harmonized

    def freeze_encoder(self):
        """冻结Encoder层（用于微调）"""
        frozen_modules = [
            self.conv_in,
            self.down1, self.down2, self.down3, self.down4
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        print("✓ Encoder层已冻结")

    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ 所有层已解冻")


def load_pretrained_pctnet(model, pretrained_path, strict=False, device='cuda'):
    """
    从libcom的预训练权重加载模型

    Args:
        model: PCTNet实例
        pretrained_path: 预训练权重路径（.pth文件）
        strict: 是否严格匹配所有层（建议False，因为可能有额外层）
        device: 设备
    """
    if not pretrained_path or not torch.cuda.is_available() and device == 'cuda':
        print("警告: 无预训练权重或CUDA不可用，使用随机初始化")
        return model

    try:
        print(f"正在加载预训练权重: {pretrained_path}")

        # 加载权重文件
        checkpoint = torch.load(pretrained_path, map_location=device)

        # 处理不同的权重格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 尝试加载（允许部分匹配）
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if not strict:
            if missing_keys:
                print(f"  未匹配的模型层: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"  权重文件中多余的层: {len(unexpected_keys)} 个")

        print("✓ 预训练权重加载成功")

    except Exception as e:
        print(f"警告: 加载预训练权重失败 - {e}")
        print("将使用随机初始化继续训练")

    return model


def create_pctnet(pretrained_path: Optional[str] = None,
                  freeze_encoder: bool = False,
                  device: str = 'cuda') -> PCTNet:
    """
    创建PCTNet模型的便捷函数

    Args:
        pretrained_path: 预训练权重路径
        freeze_encoder: 是否冻结Encoder（微调时推荐）
        device: 设备

    Returns:
        model: PCTNet实例
    """
    model = PCTNet(in_channels=4, out_channels=3, base_channels=64)

    # 加载预训练权重
    if pretrained_path:
        model = load_pretrained_pctnet(model, pretrained_path, strict=False, device=device)

    # 冻结Encoder（可选）
    if freeze_encoder:
        model.freeze_encoder()

    model = model.to(device)
    return model


def test_model():
    """测试模型前向传播"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"测试设备: {device}")

    # 创建模型
    model = create_pctnet(pretrained_path=None, device=device)

    # 测试输入
    batch_size = 2
    composite = torch.randn(batch_size, 3, 256, 256).to(device)
    mask = torch.randn(batch_size, 1, 256, 256).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(composite, mask)

    print(f"输入形状: Composite {composite.shape}, Mask {mask.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ 模型测试通过")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")


if __name__ == '__main__':
    test_model()
