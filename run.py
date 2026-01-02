"""
道路裂缝图像协调统一脚本（Composite + Harmonization）

核心特性：
1. 图像合成逻辑：100% 保留 run_harmonization.py
2. 模型加载逻辑：融合 run.py 的智能权重加载（支持官方 / 自训）
3. 支持 libcom 官方模型（pctNet / doveNet 等）
4. 支持自定义 .pth 权重
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

warnings.filterwarnings('ignore')


# ==============================
# 智能模型加载（来自 run.py，已裁剪）
# ==============================
def load_model_smart(checkpoint, model_name, model_source, device):
    """
    智能加载协调模型
    - checkpoint = None / 'libcom' → 官方预训练
    - checkpoint = *.pth           → 自定义权重
    """
    from libcom import ImageHarmonizationModel

        # ===== 自训练 ResNetUNet =====
    if model_source == 'resnet_unet':
        from mulit_models import ResNetUNet

        print("使用自训练 ResNetUNet")

        model = ResNetUNet(pretrained=False).to(device)
        model.eval()

        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']

        model.load_state_dict(state, strict=True)
        return model


    # ===== 自实现 PCTNet =====
    if model_source == 'pctnet':
        from mulit_models import create_pctnet

        print("使用自实现 PCTNet")

        model = create_pctnet(
            pretrained_path=None,
            freeze_encoder=False,
            device=device
        )
        model.eval()

        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']

        model.load_state_dict(state, strict=False)
        return model


    # libcom 期望 device: int GPU id 或 -1
    libcom_device = 0 if device.startswith('cuda') else -1

    # ---------- 官方预训练 ----------
    if checkpoint is None or checkpoint == 'libcom':
        print(f"使用 libcom 官方预训练模型: {model_name}")
        model = ImageHarmonizationModel(
            method=model_name,
            device=libcom_device
        )
        return model

    # ---------- 自定义权重 ----------
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint}")

    print(f"使用自定义权重: {checkpoint}")
    model = ImageHarmonizationModel(
        method=model_name,
        device=libcom_device
    )

    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    # ===== 关键修复：去掉 "model." 前缀 =====
    new_state = {}
    for k, v in state.items():
        if k.startswith('model.'):
            new_state[k[len('model.'):]] = v
        else:
            new_state[k] = v

    # ===== 显式加载 & 显式校验 =====
    missing, unexpected = model.model.load_state_dict(new_state, strict=False)

    print("[Checkpoint load summary]")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    if len(missing) == 0 and len(unexpected) == 0:
        print("✓ 自定义权重已 **完整正确** 加载")
    else:
        print("⚠️ 注意：权重未完全匹配，请确认模型结构或 checkpoint 来源")

    return model


# ==============================
# 裂缝协调类（完全基于 run_harmonization.py）
# ==============================
class CrackHarmonization:

    def __init__(self, model_name='pctNet', checkpoint=None, model_source='libcom', device='cuda'):
        self.model_name = model_name
        self.model_source = model_source
        self.device_str = device if torch.cuda.is_available() else 'cpu'
        self.model = load_model_smart(
            checkpoint=checkpoint,
            model_name=model_name,
            model_source=model_source,
            device=self.device_str
        )

    # ---------- 简单合成（原封不动） ----------
    def simple_composite(self, background_path, foreground_path, mask_path,
                         paste_position=None, output_path=None):

        background = Image.open(background_path).convert('RGB')
        foreground = Image.open(foreground_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if foreground.size != mask.size:
            mask = mask.resize(foreground.size, Image.LANCZOS)

        mask_array = np.array(mask)
        coords = np.argwhere(mask_array > 128)
        if len(coords) == 0:
            raise ValueError("Mask 中未检测到裂缝区域")

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        crack_region = foreground.crop((x_min, y_min, x_max + 1, y_max + 1))
        crack_mask = mask.crop((x_min, y_min, x_max + 1, y_max + 1))

        w, h = crack_region.size

        if paste_position is None:
            max_x = background.width - w - config.PASTE_MARGIN
            max_y = background.height - h - config.PASTE_MARGIN

            if max_x < config.PASTE_MARGIN or max_y < config.PASTE_MARGIN:
                print("警告: 裂缝尺寸过大，自动居中粘贴")
                paste_x = max(0, (background.width - w) // 2)
                paste_y = max(0, (background.height - h) // 2)
            else:
                paste_x = random.randint(config.PASTE_MARGIN, max_x)
                paste_y = random.randint(config.PASTE_MARGIN, max_y)

        else:
            paste_x, paste_y = paste_position

        full_mask = Image.new('L', background.size, 0)
        full_mask.paste(crack_mask, (paste_x, paste_y))

        composite = background.copy()
        composite.paste(crack_region, (paste_x, paste_y), crack_mask)

        if output_path:
            composite.save(output_path, quality=config.OUTPUT_QUALITY)

        return composite, full_mask, (paste_x, paste_y)

    # ---------- AI 协调（原逻辑） ----------
    def harmonize(self, composite_img, mask_img, output_path=None):

        composite_np = np.array(composite_img)
        mask_np = np.array(mask_img)

        if self.model_source == 'libcom':
            harmonized_np = self.model(composite_np, mask_np)
        else:
            device = next(self.model.parameters()).device

            composite_t = torch.from_numpy(composite_np).permute(2,0,1).unsqueeze(0).float().to(device) / 255.
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device) / 255.

            with torch.no_grad():
                out = self.model(composite_t, mask_t)

            harmonized_np = (
                out.squeeze(0)
                .permute(1,2,0)
                .clamp(0,1)
                .cpu()
                .numpy() * 255
            ).astype(np.uint8)

        harmonized_img = Image.fromarray(harmonized_np.astype(np.uint8))

        if output_path:
            harmonized_img.save(output_path, quality=config.OUTPUT_QUALITY)

        return harmonized_img

    # ---------- 可视化 ----------
    def visualize(self, bg_path, comp, harm, save_path):

        bg = Image.open(bg_path).convert('RGB')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(bg); axes[0].set_title("Background"); axes[0].axis('off')
        axes[1].imshow(comp); axes[1].set_title("Naive Composite"); axes[1].axis('off')
        axes[2].imshow(harm); axes[2].set_title("Harmonized"); axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


# ==============================
# 主函数
# ==============================
def main():

    parser = argparse.ArgumentParser("Crack Harmonization (Unified)")
    parser.add_argument('-b', '--background', required=False, default='input/2.jpg',)
    parser.add_argument('-f', '--foreground', required=False, default='input/kengcao.jpg',)
    parser.add_argument('-m', '--mask', required=False, default='input/kengcao.png',)

    parser.add_argument('--model',
                        default=config.HARMONIZATION_MODEL,
                        choices=['pctNet', 'doveNet', 'rainNet', 'htNet',
                                 'dccNet', 'bctNet', 'iscNet'])
    
    parser.add_argument('--model_source',
                        default='libcom',
                        choices=['libcom', 'resnet_unet', 'pctnet'],
                        help='模型来源：libcom 官方 / 自训练 ResNetUNet / 自实现 PCTNet')


    parser.add_argument('--checkpoint',
                        default=None,
                        help="None/libcom 使用官方权重，或指定 .pth")

    parser.add_argument('--device', default=config.DEVICE)
    args = parser.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    harmonizer = CrackHarmonization(
        model_name=args.model,
        checkpoint=args.checkpoint,
        model_source=args.model_source,
        device=args.device
    )

    name = Path(args.foreground).stem
    naive_path = os.path.join(config.OUTPUT_DIR, f'{name}_naive.jpg')
    harm_path = os.path.join(config.OUTPUT_DIR, f'{name}_harmonized.jpg')
    vis_path = os.path.join(config.OUTPUT_DIR, f'{name}_compare.png')

    comp, mask, _ = harmonizer.simple_composite(
        args.background, args.foreground, args.mask, output_path=naive_path
    )

    harm = harmonizer.harmonize(comp, mask, output_path=harm_path)

    harmonizer.visualize(args.background, comp, harm, vis_path)

    print("\n✓ 完成")
    print(f"Naive: {naive_path}")
    print(f"Harmonized: {harm_path}")
    print(f"Compare: {vis_path}")


if __name__ == '__main__':
    main()
