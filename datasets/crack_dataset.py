"""
道路裂缝协调数据集 - 在线合成版本
核心思路：从真实图片生成"不协调的合成图"作为训练输入

数据流:
    真实图片 (Real) + Mask
        → 数据增强 (仅在Mask区域)
        → 生成不协调合成图 (Composite)
        → [模型训练] Composite → Real
"""
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2


class CrackHarmonizationDataset(Dataset):
    """
    道路裂缝图像协调数据集（在线合成）

    关键特性:
    1. 不需要预先准备合成图，实时生成
    2. 通过数据增强制造"不协调感"
    3. 真实图作为Ground Truth
    """

    def __init__(self,
                 image_dir,
                 mask_dir,
                 image_size=256,
                 mode='train',
                 augmentation_intensity='medium'):
        """
        Args:
            image_dir: 真实道路裂缝图片目录
            mask_dir: 对应的掩码目录（白色=裂缝区域）
            image_size: 输出图像尺寸
            mode: 'train' or 'val'
            augmentation_intensity: 数据增强强度 ('low', 'medium', 'high')
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.mode = mode
        self.aug_intensity = augmentation_intensity

        # 获取所有图片文件
        self.image_files = self._get_image_files(image_dir)
        self.mask_files = self._get_image_files(mask_dir)

        # 文件名匹配检查
        assert len(self.image_files) > 0, f"未在 {image_dir} 找到图片文件"
        assert len(self.mask_files) > 0, f"未在 {mask_dir} 找到掩码文件"

        print(f"[{mode}] 加载了 {len(self.image_files)} 张图片, {len(self.mask_files)} 张掩码")

        # 增强参数配置
        self.aug_params = self._get_augmentation_params(augmentation_intensity)

    def _get_image_files(self, directory):
        """获取目录下的所有图片文件"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        for fname in sorted(os.listdir(directory)):
            if any(fname.lower().endswith(ext) for ext in valid_extensions):
                files.append(os.path.join(directory, fname))
        return files

    def _get_augmentation_params(self, intensity):
        """根据强度返回增强参数范围"""
        params = {
            'low': {
                'brightness': (0.9, 1.1),
                'contrast': (0.9, 1.1),
                'saturation': (0.9, 1.1),
                'hue': (-0.05, 0.05),
                'gamma': (0.95, 1.05),
            },
            'medium': {
                'brightness': (0.7, 1.3),
                'contrast': (0.7, 1.3),
                'saturation': (0.6, 1.4),
                'hue': (-0.1, 0.1),
                'gamma': (0.8, 1.2),
            },
            'high': {
                'brightness': (0.5, 1.5),
                'contrast': (0.5, 1.5),
                'saturation': (0.4, 1.6),
                'hue': (-0.2, 0.2),
                'gamma': (0.6, 1.4),
            }
        }
        return params.get(intensity, params['medium'])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        核心方法: 在线生成训练样本

        Returns:
            composite: 不协调的合成图 (输入)
            real: 真实图 (Ground Truth)
            mask: 掩码 (协调区域)
        """
        # 1. 读取真实图片和掩码
        image_path = self.image_files[idx]
        # 简单匹配：假设掩码文件名与图片相同（或有_mask后缀）
        mask_path = self._find_matching_mask(image_path)

        real_image = Image.open(image_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')

        # 2. Resize到统一尺寸
        real_image = real_image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_image = mask_image.resize((self.image_size, self.image_size), Image.NEAREST)

        # 3. 【核心】在线生成不协调的合成图
        composite_image = self._generate_disharmony(real_image, mask_image)

        # 4. 数据增强（训练时）
        if self.mode == 'train':
            composite_image, real_image, mask_image = self._apply_augmentation(
                composite_image, real_image, mask_image
            )

        # 5. 转换为Tensor
        composite = TF.to_tensor(composite_image)
        real = TF.to_tensor(real_image)
        mask = TF.to_tensor(mask_image)

        # 归一化到 [-1, 1]
        composite = composite * 2 - 1
        real = real * 2 - 1

        return {
            'composite': composite,  # 输入: 不协调图
            'real': real,           # 目标: 真实图
            'mask': mask,           # 掩码: 需要协调的区域
            'image_path': image_path
        }

    def _find_matching_mask(self, image_path):
        """根据图片路径查找对应的掩码文件"""
        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]

        # 尝试多种命名模式
        possible_names = [
            basename,  # 完全相同
            name_without_ext + '_mask.jpg',
            name_without_ext + '_mask.png',
            name_without_ext + '.png',
            name_without_ext + '.jpg',
        ]

        for possible_name in possible_names:
            mask_path = os.path.join(self.mask_dir, possible_name)
            if os.path.exists(mask_path):
                return mask_path

        # 如果都不存在，返回第一个掩码（调试用）
        # 实际项目中应该抛出异常
        print(f"警告: 未找到匹配的掩码文件 {basename}, 使用第一个掩码")
        return self.mask_files[0]

    def _generate_disharmony(self, real_image, mask_image):
        """
        核心方法: 生成不协调的合成图

        策略: 仅在Mask区域内应用颜色/亮度变换，制造不协调感

        实现方式:
        1. LUT (Look-Up Table) 变换 (70%概率)
        2. HSV随机抖动 (30%概率)
        """
        real_np = np.array(real_image)
        mask_np = np.array(mask_image)

        # 二值化mask (>128为前景区域)
        mask_binary = (mask_np > 128).astype(np.uint8)

        # 随机选择增强方法
        if random.random() < 0.7:
            # 方法1: LUT变换（推荐）
            composite_np = self._apply_lut_transform(real_np, mask_binary)
        else:
            # 方法2: HSV变换
            composite_np = self._apply_hsv_transform(real_np, mask_binary)

        # 可选: 添加轻微噪声（增加真实感）
        if random.random() < 0.3:
            composite_np = self._add_noise(composite_np, mask_binary)

        return Image.fromarray(composite_np.astype(np.uint8))

    def _apply_lut_transform(self, image, mask):
        """
        LUT (Look-Up Table) 颜色变换
        原理: 构建随机的映射曲线，改变像素值分布
        """
        h, w, c = image.shape
        composite = image.copy()

        # 为每个通道生成随机LUT曲线
        for channel in range(3):
            # 生成随机控制点
            anchor_points = np.array([0, 64, 128, 192, 255])

            # 根据增强强度随机偏移
            brightness_factor = random.uniform(*self.aug_params['brightness'])
            offsets = np.random.randint(-30, 30, size=5) * (brightness_factor - 1.0)
            new_values = np.clip(anchor_points + offsets, 0, 255)

            # 插值生成完整LUT
            lut = np.interp(np.arange(256), anchor_points, new_values).astype(np.uint8)

            # 仅在Mask区域应用LUT
            channel_data = composite[:, :, channel]
            transformed = cv2.LUT(channel_data, lut)
            composite[:, :, channel] = np.where(mask > 0, transformed, channel_data)

        return composite

    def _apply_hsv_transform(self, image, mask):
        """
        HSV颜色空间变换
        原理: 在HSV空间中随机调整色调、饱和度、亮度
        """
        composite = image.copy()

        # 转换到HSV空间
        hsv = cv2.cvtColor(composite, cv2.COLOR_RGB2HSV).astype(np.float32)

        # 随机调整参数
        h_shift = random.uniform(*self.aug_params['hue']) * 180  # 色调偏移
        s_scale = random.uniform(*self.aug_params['saturation'])  # 饱和度缩放
        v_scale = random.uniform(*self.aug_params['brightness'])  # 亮度缩放

        # 仅在Mask区域应用变换
        mask_3d = np.stack([mask] * 3, axis=-1)

        hsv[:, :, 0] = np.where(mask > 0, (hsv[:, :, 0] + h_shift) % 180, hsv[:, :, 0])
        hsv[:, :, 1] = np.where(mask > 0, np.clip(hsv[:, :, 1] * s_scale, 0, 255), hsv[:, :, 1])
        hsv[:, :, 2] = np.where(mask > 0, np.clip(hsv[:, :, 2] * v_scale, 0, 255), hsv[:, :, 2])

        # 转换回RGB
        composite = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return composite

    def _add_noise(self, image, mask, noise_strength=5):
        """在Mask区域添加轻微噪声"""
        noise = np.random.normal(0, noise_strength, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255)

        # 仅在Mask区域应用
        mask_3d = np.stack([mask] * 3, axis=-1)
        return np.where(mask_3d > 0, noisy, image).astype(np.uint8)

    def _apply_augmentation(self, composite, real, mask):
        """
        数据增强（训练时）
        注意: 这里的增强是对整个图像的，不是制造不协调
        """
        # 随机水平翻转
        if random.random() < 0.5:
            composite = TF.hflip(composite)
            real = TF.hflip(real)
            mask = TF.hflip(mask)

        # 随机旋转（小角度）
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            composite = TF.rotate(composite, angle)
            real = TF.rotate(real, angle)
            mask = TF.rotate(mask, angle)

        return composite, real, mask


def test_dataset():
    """测试数据集功能"""
    import matplotlib.pyplot as plt

    # 假设你的数据路径
    image_dir = "/home/enine/FXM/DIH/data/foregrounds"  # 替换为你的路径
    mask_dir = "/home/enine/FXM/DIH/data/masks"

    dataset = CrackHarmonizationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=256,
        mode='train',
        augmentation_intensity='medium'
    )

    # 测试单个样本
    sample = dataset[0]

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 反归一化用于显示
    composite_show = (sample['composite'].permute(1, 2, 0).numpy() + 1) / 2
    real_show = (sample['real'].permute(1, 2, 0).numpy() + 1) / 2
    mask_show = sample['mask'].squeeze().numpy()

    axes[0].imshow(np.clip(composite_show, 0, 1))
    axes[0].set_title('Composite (Input)')
    axes[0].axis('off')

    axes[1].imshow(np.clip(real_show, 0, 1))
    axes[1].set_title('Real (GT)')
    axes[1].axis('off')

    axes[2].imshow(mask_show, cmap='gray')
    axes[2].set_title('Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_test.png', dpi=150)
    print("测试图片已保存: dataset_test.png")


if __name__ == '__main__':
    test_dataset()
