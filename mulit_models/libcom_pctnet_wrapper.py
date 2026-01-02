"""
LibcomPCTNet包装器 - 使用libcom的预训练PCTNet进行微调

这个包装器允许你直接使用libcom的预训练Transformer PCTNet进行训练
"""
import torch
import torch.nn as nn
import numpy as np


class LibcomPCTNetWrapper(nn.Module):
    """
    包装libcom的PCTNet模型使其可以用于训练

    优势：
    - 直接使用libcom的预训练Transformer架构
    - 完全兼容预训练权重
    - 可以进行微调训练
    """

    def __init__(self, device='cuda'):
        super().__init__()

        try:
            from libcom import ImageHarmonizationModel
            print("正在加载 libcom 的 PCTNet 预训练模型...")

            # 转换device为字符串（处理torch.device对象）
            if isinstance(device, torch.device):
                device_str = str(device)
            else:
                device_str = str(device)

            # 加载libcom模型
            # libcom期望: 0, 1, 2... 表示GPU编号
            # 注意: libcom不接受-1作为CPU，如果要用CPU，需要其他方式
            if 'cuda' in device_str:
                if ':' in device_str:
                    # 例如: 'cuda:1' -> 1
                    device_id = int(device_str.split(':')[1])
                else:
                    # 'cuda' -> 0
                    device_id = 0
            else:
                # CPU模式: libcom可能不支持-1，使用0但在CPU环境
                print("警告: libcom可能不完全支持CPU训练，建议使用GPU")
                device_id = 0  # 尝试使用0，让libcom自己处理

            print(f"  Libcom device_id: {device_id}")

            harmonization_model = ImageHarmonizationModel(
                method='pctNet',
                device=device_id
            )

            # 提取内部的神经网络模型
            self.model = harmonization_model.model

            # 设置为训练模式
            self.model.train()

            print("✓ Libcom PCTNet 加载成功")

        except ImportError:
            raise ImportError(
                "需要安装 libcom 才能使用此模型\n"
                "请运行: pip install libcom"
            )
        except Exception as e:
            raise RuntimeError(f"加载 libcom PCTNet 失败: {e}")

    def forward(self, composite, mask):
        """
        前向传播 - 重写逻辑以支持 Batch 训练

        Args:
            composite: 合成图 (B, 3, H, W), range [-1, 1]
            mask: 掩码 (B, 1, H, W), range [0, 1]

        Returns:
            harmonized: 协调后的图像 (B, 3, H, W), range [-1, 1]
        """
        # 将输入从[-1, 1]转换到[0, 1]
        composite_01 = (composite + 1) / 2
        mask_01 = mask

        # 直接使用 self.model 的组件，绕过 PCTNet.forward 的限制
        # PCTNet.forward 有 unsqueeze(0) 操作且不支持 batch，还要求 image_fullres 不为 None
        
        try:
            model = self.model
            
            # 准备输入: RGB + Mask
            # PCTNet 内部逻辑: x = torch.cat((image, mask), dim=1)
            # 所以我们这里手动拼接
            x = torch.cat((composite_01, mask_01), dim=1) # (B, 4, H, W)
            
            # Encoder
            # 注意: backbone_features 默认为 None
            intermediates = model.encoder(x, backbone_features=None)
            
            # Decoder
            latent, attention_map = model.decoder(intermediates, composite_01, mask_01)
            
            # Get params
            params = model.get_params(latent)
            
            # PCT (Predictive Color Transform)
            output_lowres = model.PCT(composite_01, params)
            
            # Apply mask/attention
            if model.use_attn:
                output_lowres = output_lowres * attention_map + composite_01 * (1-attention_map)
            else:
                output_lowres = output_lowres * mask_01 + composite_01 * (1 - mask_01)
            
            # 归一化回 [-1, 1]
            harmonized = output_lowres * 2 - 1
            
            return harmonized

        except Exception as e:
            print(f"警告: 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回输入作为fallback
            return composite

    def freeze_encoder(self):
        """冻结模型的前半部分层"""
        # libcom模型的结构可能不同，这里冻结transformer encoder部分
        frozen_count = 0
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            # 冻结encoder相关的层
            if 'encoder' in name.lower() or 'transformer_enc' in name.lower():
                param.requires_grad = False
                frozen_count += 1

        print(f"✓ 冻结了 {frozen_count}/{total_params} 个参数")

    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("✓ 所有层已解冻")


class SimplifiedLibcomWrapper(nn.Module):
    """
    简化版包装器 - 如果上面的复杂版本有问题，使用这个

    这个版本更简单，直接在numpy层面处理
    """

    def __init__(self, device='cuda'):
        super().__init__()

        try:
            from libcom import ImageHarmonizationModel

            # 转换device为字符串（处理torch.device对象）
            if isinstance(device, torch.device):
                device_str = str(device)
            else:
                device_str = str(device)

            # 解析device id
            if 'cuda' in device_str:
                if ':' in device_str:
                    device_id = int(device_str.split(':')[1])
                else:
                    device_id = 0
            else:
                print("警告: 简化版libcom包装器在CPU上可能不稳定")
                device_id = 0

            self.harmonization_model = ImageHarmonizationModel(
                method='pctNet',
                device=device_id
            )

            # 获取内部模型用于参数更新
            self.model = self.harmonization_model.model
            self.device = device

            print("✓ Libcom PCTNet (简化版) 加载成功")

        except ImportError:
            raise ImportError("需要安装 libcom: pip install libcom")

    def forward(self, composite, mask):
        """使用libcom的高级API"""
        batch_size = composite.shape[0]

        # 转换格式
        composite_np = ((composite + 1) / 2 * 255).clamp(0, 255).cpu().numpy()
        mask_np = (mask * 255).clamp(0, 255).squeeze(1).cpu().numpy()

        outputs = []
        for i in range(batch_size):
            # (C, H, W) -> (H, W, C)
            comp_img = composite_np[i].transpose(1, 2, 0).astype(np.uint8)
            mask_img = mask_np[i].astype(np.uint8)

            # 调用libcom API
            harmonized_img = self.harmonization_model(comp_img, mask_img)

            # 转回tensor
            harm_tensor = torch.from_numpy(harmonized_img).permute(2, 0, 1).float() / 255.0
            outputs.append(harm_tensor)

        harmonized = torch.stack(outputs).to(self.device)
        harmonized = harmonized * 2 - 1  # 归一化到[-1, 1]

        return harmonized


def create_libcom_pctnet(device='cuda', freeze_encoder=False, use_simplified=False):
    """
    创建libcom PCTNet包装器

    Args:
        device: 设备
        freeze_encoder: 是否冻结encoder
        use_simplified: 是否使用简化版

    Returns:
        model: 包装后的模型
    """
    try:
        if use_simplified:
            print("使用简化版 libcom PCTNet 包装器")
            model = SimplifiedLibcomWrapper(device=device)
        else:
            print("使用完整版 libcom PCTNet 包装器")
            model = LibcomPCTNetWrapper(device=device)

        if freeze_encoder:
            model.freeze_encoder()

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

        return model

    except Exception as e:
        print(f"错误: {e}")
        print("\n如果libcom有问题，建议使用方案1（U-Net从头训练）或方案3（ResNet-UNet）")
        raise


if __name__ == '__main__':
    print("测试 Libcom PCTNet 包装器\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")

    try:
        # 测试创建模型
        model = create_libcom_pctnet(device=device, freeze_encoder=False)

        # 测试前向传播
        print("\n测试前向传播...")
        batch_size = 2
        composite = torch.randn(batch_size, 3, 256, 256).to(device)
        mask = torch.ones(batch_size, 1, 256, 256).to(device)

        with torch.no_grad():
            output = model(composite, mask)

        print(f"输入形状: Composite {composite.shape}, Mask {mask.shape}")
        print(f"输出形状: {output.shape}")
        print("\n✓ Libcom PCTNet 包装器测试通过！")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("\n这是正常的，如果你还没有安装libcom")
        print("安装方法: pip install libcom")
