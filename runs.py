"""
道路裂缝图像批量协调统一脚本（Batch Composite + Harmonization）

功能：
- 将 foreground 文件夹中的病害图像批量融合到 background 文件夹中的干净路面上
- 生成两个文件夹：harmonized（合成图像）和 masks（对应mask）
- 支持直接用于语义分割模型训练
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import random
import warnings
from pathlib import Path
from tqdm import tqdm
import glob

import numpy as np
from PIL import Image
import torch

import config

warnings.filterwarnings('ignore')


# ==============================
# 智能模型加载
# ==============================
def load_model_smart(checkpoint, model_name, model_source, device):
    """
    智能加载协调模型
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
        model = create_pctnet(pretrained_path=None, freeze_encoder=False, device=device)
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
        model = ImageHarmonizationModel(method=model_name, device=libcom_device)
        return model

    # ---------- 自定义权重 ----------
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint}")

    print(f"使用自定义权重: {checkpoint}")
    model = ImageHarmonizationModel(method=model_name, device=libcom_device)

    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    new_state = {}
    for k, v in state.items():
        if k.startswith('model.'):
            new_state[k[len('model.'):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.model.load_state_dict(new_state, strict=False)
    print(f"[Checkpoint] Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return model


# ==============================
# 批量裂缝协调类
# ==============================
class BatchCrackHarmonization:

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

    def simple_composite(self, background, foreground, mask, paste_position=None):
        """
        简单合成（支持PIL Image或路径输入）
        返回：composite, full_mask, position
        """
        # 处理输入类型
        if isinstance(background, (str, Path)):
            background = Image.open(background).convert('RGB')
        if isinstance(foreground, (str, Path)):
            foreground = Image.open(foreground).convert('RGB')
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask).convert('L')

        if foreground.size != mask.size:
            mask = mask.resize(foreground.size, Image.LANCZOS)

        mask_array = np.array(mask)
        coords = np.argwhere(mask_array > 128)
        if len(coords) == 0:
            raise ValueError("Mask 中未检测到病害区域")

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        crack_region = foreground.crop((x_min, y_min, x_max + 1, y_max + 1))
        crack_mask = mask.crop((x_min, y_min, x_max + 1, y_max + 1))

        w, h = crack_region.size

        if paste_position is None:
            margin = getattr(config, 'PASTE_MARGIN', 10)
            max_x = background.width - w - margin
            max_y = background.height - h - margin

            if max_x < margin or max_y < margin:
                paste_x = max(0, (background.width - w) // 2)
                paste_y = max(0, (background.height - h) // 2)
            else:
                paste_x = random.randint(margin, max_x)
                paste_y = random.randint(margin, max_y)
        else:
            paste_x, paste_y = paste_position

        full_mask = Image.new('L', background.size, 0)
        full_mask.paste(crack_mask, (paste_x, paste_y))

        composite = background.copy()
        composite.paste(crack_region, (paste_x, paste_y), crack_mask)

        return composite, full_mask, (paste_x, paste_y)

    def harmonize(self, composite_img, mask_img):
        """
        AI 协调
        """
        composite_np = np.array(composite_img)
        mask_np = np.array(mask_img)

        if self.model_source == 'libcom':
            harmonized_np = self.model(composite_np, mask_np)
        else:
            device = next(self.model.parameters()).device
            composite_t = torch.from_numpy(composite_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device) / 255.

            with torch.no_grad():
                out = self.model(composite_t, mask_t)

            harmonized_np = (
                out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)

        harmonized_img = Image.fromarray(harmonized_np.astype(np.uint8))
        return harmonized_img


def get_image_files(folder, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
    """获取文件夹中的所有图像文件"""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f'*{ext}')))
        files.extend(glob.glob(os.path.join(folder, f'*{ext.upper()}')))
    return sorted(files)


def find_matching_mask(foreground_path, mask_folder):
    """
    根据前景图像路径找到对应的mask
    支持多种命名规则：
    1. 同名不同后缀：fg.jpg -> fg.png
    2. 添加_mask后缀：fg.jpg -> fg_mask.png
    """
    fg_stem = Path(foreground_path).stem
    
    # 尝试多种匹配模式
    patterns = [
        os.path.join(mask_folder, f'{fg_stem}.png'),
        os.path.join(mask_folder, f'{fg_stem}.jpg'),
        os.path.join(mask_folder, f'{fg_stem}_mask.png'),
        os.path.join(mask_folder, f'{fg_stem}_mask.jpg'),
        os.path.join(mask_folder, f'{fg_stem}.*'),
    ]
    
    for pattern in patterns[:-1]:  # 精确匹配
        if os.path.exists(pattern):
            return pattern
    
    # 模糊匹配
    matches = glob.glob(patterns[-1])
    if matches:
        return matches[0]
    
    return None


def batch_process(args):
    """
    批量处理主函数
    """
    # 创建输出文件夹
    output_harmonized = os.path.join(args.output, 'harmonized')
    output_masks = os.path.join(args.output, 'masks')
    os.makedirs(output_harmonized, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)

    # 获取所有文件
    background_files = get_image_files(args.background_folder)
    foreground_files = get_image_files(args.foreground_folder)

    print(f"\n{'='*60}")
    print(f"批量处理配置")
    print(f"{'='*60}")
    print(f"背景图像文件夹: {args.background_folder}")
    print(f"前景图像文件夹: {args.foreground_folder}")
    print(f"Mask文件夹: {args.mask_folder}")
    print(f"输出文件夹: {args.output}")
    print(f"{'='*60}")
    print(f"背景图像数量: {len(background_files)}")
    print(f"前景图像数量: {len(foreground_files)}")
    print(f"预计生成图像: {len(background_files) * len(foreground_files)} 对")
    print(f"{'='*60}\n")

    if len(background_files) == 0:
        raise ValueError(f"背景文件夹为空: {args.background_folder}")
    if len(foreground_files) == 0:
        raise ValueError(f"前景文件夹为空: {args.foreground_folder}")

    # 初始化模型
    print("正在加载模型...")
    harmonizer = BatchCrackHarmonization(
        model_name=args.model,
        checkpoint=args.checkpoint,
        model_source=args.model_source,
        device=args.device
    )
    print("模型加载完成！\n")

    # 统计
    success_count = 0
    fail_count = 0
    failed_items = []

    # 计算总任务数
    total_tasks = len(background_files) * len(foreground_files)
    
    # 使用 tqdm 显示进度
    pbar = tqdm(total=total_tasks, desc="批量处理进度")

    for bg_idx, bg_path in enumerate(background_files):
        bg_name = Path(bg_path).stem
        
        # 预加载背景图像（避免重复读取）
        try:
            background_img = Image.open(bg_path).convert('RGB')
        except Exception as e:
            print(f"\n⚠️ 无法加载背景图像: {bg_path}, 错误: {e}")
            fail_count += len(foreground_files)
            pbar.update(len(foreground_files))
            continue

        for fg_idx, fg_path in enumerate(foreground_files):
            fg_name = Path(fg_path).stem
            
            try:
                # 查找对应的mask
                mask_path = find_matching_mask(fg_path, args.mask_folder)
                if mask_path is None:
                    raise FileNotFoundError(f"找不到对应的mask: {fg_path}")

                # 加载前景和mask
                foreground_img = Image.open(fg_path).convert('RGB')
                mask_img = Image.open(mask_path).convert('L')

                # 合成
                composite, full_mask, position = harmonizer.simple_composite(
                    background_img.copy(),  # 使用副本
                    foreground_img,
                    mask_img,
                    paste_position=None  # 随机位置
                )

                # 协调
                harmonized = harmonizer.harmonize(composite, full_mask)

                # 生成文件名
                # 格式：bg{背景编号}_fg{前景编号}_{前景名称}.jpg/png
                out_name = f"bg{bg_idx:03d}_fg{fg_idx:03d}_{fg_name}"
                
                # 保存协调后的图像
                harmonized_path = os.path.join(output_harmonized, f"{out_name}.jpg")
                harmonized.save(harmonized_path, quality=95)

                # 保存mask（保持为PNG以确保无损）
                mask_out_path = os.path.join(output_masks, f"{out_name}.png")
                full_mask.save(mask_out_path)

                success_count += 1

            except Exception as e:
                fail_count += 1
                failed_items.append({
                    'background': bg_path,
                    'foreground': fg_path,
                    'error': str(e)
                })
                if args.verbose:
                    tqdm.write(f"⚠️ 处理失败: bg={bg_name}, fg={fg_name}, 错误: {e}")

            pbar.update(1)

    pbar.close()

    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"处理完成统计")
    print(f"{'='*60}")
    print(f"✓ 成功: {success_count}")
    print(f"✗ 失败: {fail_count}")
    print(f"{'='*60}")
    print(f"输出路径:")
    print(f"  - 协调图像: {output_harmonized}")
    print(f"  - Mask图像: {output_masks}")
    print(f"{'='*60}")

    # 如果有失败的，保存日志
    if failed_items:
        log_path = os.path.join(args.output, 'failed_log.txt')
        with open(log_path, 'w') as f:
            for item in failed_items:
                f.write(f"BG: {item['background']}\n")
                f.write(f"FG: {item['foreground']}\n")
                f.write(f"Error: {item['error']}\n")
                f.write("-" * 40 + "\n")
        print(f"失败日志已保存: {log_path}")

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="批量道路病害图像协调合成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python batch_harmonization.py -b ./backgrounds -f ./foregrounds -m ./masks -o ./output
  
文件夹结构示例:
  backgrounds/
    ├── road_001.jpg
    ├── road_002.jpg
    └── ...
  
  foregrounds/
    ├── crack_001.jpg
    ├── crack_002.jpg
    └── ...
  
  masks/
    ├── crack_001.png  (与foreground同名)
    ├── crack_002.png
    └── ...
  
输出结构:
  output/
    ├── harmonized/
    │   ├── bg000_fg000_crack_001.jpg
    │   ├── bg000_fg001_crack_002.jpg
    │   └── ...
    └── masks/
        ├── bg000_fg000_crack_001.png
        ├── bg000_fg001_crack_002.png
        └── ...
        """
    )

    # 输入文件夹
    parser.add_argument('-b', '--background_folder', required=False, default='input_piliang/clean',
                        help='干净路面背景图像文件夹')
    parser.add_argument('-f', '--foreground_folder', required=False, default='input_piliang/die',
                        help='病害前景图像文件夹')
    parser.add_argument('-m', '--mask_folder', required=False, default='input_piliang/mask',
                        help='病害mask文件夹（与foreground对应）')
    
    # 输出文件夹
    parser.add_argument('-o', '--output', default='./batch_output',
                        help='输出文件夹（默认: ./batch_output）')

    # 模型配置
    parser.add_argument('--model', default='pctNet',
                        choices=['pctNet', 'doveNet', 'rainNet', 'htNet',
                                 'dccNet', 'bctNet', 'iscNet'],
                        help='协调模型名称')
    parser.add_argument('--model_source', default='libcom',
                        choices=['libcom', 'resnet_unet', 'pctnet'],
                        help='模型来源')
    parser.add_argument('--checkpoint', default=None,
                        help='自定义权重路径（None使用官方预训练）')
    parser.add_argument('--device', default='cuda',
                        help='计算设备（cuda/cpu）')

    # 其他选项
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（控制粘贴位置的随机性）')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细错误信息')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 执行批量处理
    batch_process(args)


if __name__ == '__main__':
    main()