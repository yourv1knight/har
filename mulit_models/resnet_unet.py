"""
ResNet-UNet: 使用ImageNet预训练的ResNet作为Encoder
这样可以利用预训练权重，且训练速度更快
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """使用预训练ResNet作为Encoder"""

    def __init__(self, pretrained=True):
        super().__init__()

        # 加载预训练的ResNet18（轻量级）
        resnet = models.resnet18(pretrained=pretrained)

        # 修改第一层以接受4通道输入（RGB + Mask）
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 如果使用预训练，复制RGB的权重并为Mask通道随机初始化
        if pretrained:
            pretrained_weight = resnet.conv1.weight.data
            self.conv1.weight.data[:, :3, :, :] = pretrained_weight
            self.conv1.weight.data[:, 3:, :, :] = pretrained_weight.mean(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet的各个block
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    def forward(self, x):
        """返回各级特征用于skip connection"""
        x0 = self.relu(self.bn1(self.conv1(x)))  # 64, H/2, W/2
        x1 = self.maxpool(x0)                     # 64, H/4, W/4

        x2 = self.layer1(x1)                      # 64, H/4, W/4
        x3 = self.layer2(x2)                      # 128, H/8, W/8
        x4 = self.layer3(x3)                      # 256, H/16, W/16
        x5 = self.layer4(x4)                      # 512, H/32, W/32

        return x5, x4, x3, x2, x0


class DecoderBlock(nn.Module):
    """Decoder上采样块"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode='bilinear',
                align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ResNetUNet(nn.Module):
    """
    ResNet-UNet: 使用预训练ResNet18作为Encoder

    优势：
    - 使用ImageNet预训练权重（大规模数据集）
    - 训练更快，收敛更好
    - 比从头训练效果更好
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Encoder (ResNet18, 预训练)
        self.encoder = ResNetEncoder(pretrained=pretrained)

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)  # 512+256 -> 256
        self.decoder3 = DecoderBlock(256, 128, 128)  # 256+128 -> 128
        self.decoder2 = DecoderBlock(128, 64, 64)    # 128+64 -> 64
        self.decoder1 = DecoderBlock(64, 64, 64)     # 64+64 -> 64

        # 最后的上采样
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # 输出层
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, composite, mask):
        """
        Args:
            composite: (B, 3, H, W)
            mask: (B, 1, H, W)
        Returns:
            harmonized: (B, 3, H, W)
        """
        # 拼接输入
        x = torch.cat([composite, mask], dim=1)  # (B, 4, H, W)

        # Encoder
        x5, x4, x3, x2, x0 = self.encoder(x)

        # Decoder with skip connections
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x0)

        # 最终上采样到原始分辨率
        d0 = self.final_up(d1)

        # 输出
        out = self.out_conv(d0)
        if out.size(2) != composite.size(2) or out.size(3) != composite.size(3):
            out = F.interpolate(
                out,
                size=(composite.size(2), composite.size(3)),
                mode='bilinear',
                align_corners=False
            )
        # 残差连接
        harmonized = composite + out

        return harmonized

    def freeze_encoder(self):
        """冻结Encoder（ResNet部分）"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ ResNet Encoder已冻结")

    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ 所有层已解冻")


def create_resnet_unet(pretrained=True, freeze_encoder=False, device='cuda'):
    """
    创建ResNet-UNet模型

    Args:
        pretrained: 是否使用ImageNet预训练权重
        freeze_encoder: 是否冻结Encoder
        device: 设备

    Returns:
        model: ResNet-UNet实例
    """
    print(f"创建 ResNet-UNet (pretrained={pretrained})")

    model = ResNetUNet(pretrained=pretrained)

    if freeze_encoder:
        model.freeze_encoder()

    model = model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    return model


if __name__ == '__main__':
    # 测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"测试设备: {device}\n")

    # 创建模型
    model = create_resnet_unet(pretrained=True, freeze_encoder=False, device=device)

    # 测试前向传播
    batch_size = 2
    composite = torch.randn(batch_size, 3, 256, 256).to(device)
    mask = torch.randn(batch_size, 1, 256, 256).to(device)

    with torch.no_grad():
        output = model(composite, mask)

    print(f"\n输入形状: Composite {composite.shape}, Mask {mask.shape}")
    print(f"输出形状: {output.shape}")
    print("\n✓ ResNet-UNet测试通过！")
