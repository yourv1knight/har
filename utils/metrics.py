"""
评估指标计算工具
用于评估图像协调效果
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2, data_range=1.0):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)

    Args:
        img1: 图片1 (numpy array or torch tensor)
        img2: 图片2 (numpy array or torch tensor)
        data_range: 数据范围（默认1.0，即[0,1]范围）

    Returns:
        psnr_value: PSNR值
    """
    # 转为numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # 如果是batch，计算平均
    if len(img1.shape) == 4:
        psnr_values = []
        for i in range(img1.shape[0]):
            # (C, H, W) -> (H, W, C)
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            psnr_values.append(psnr(im1, im2, data_range=data_range))
        return np.mean(psnr_values)
    else:
        # 单张图片
        if len(img1.shape) == 3 and img1.shape[0] == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        return psnr(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, data_range=1.0):
    """
    计算SSIM (Structural Similarity Index)

    Args:
        img1: 图片1
        img2: 图片2
        data_range: 数据范围

    Returns:
        ssim_value: SSIM值
    """
    # 转为numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # 如果是batch，计算平均
    if len(img1.shape) == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            # (C, H, W) -> (H, W, C)
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            ssim_values.append(ssim(im1, im2, data_range=data_range, channel_axis=2))
        return np.mean(ssim_values)
    else:
        # 单张图片
        if len(img1.shape) == 3 and img1.shape[0] == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        return ssim(img1, img2, data_range=data_range, channel_axis=2)


def calculate_mae(img1, img2):
    """
    计算MAE (Mean Absolute Error)

    Args:
        img1: 图片1
        img2: 图片2

    Returns:
        mae_value: MAE值
    """
    if isinstance(img1, torch.Tensor):
        return torch.mean(torch.abs(img1 - img2)).item()
    else:
        return np.mean(np.abs(img1 - img2))


def calculate_mse(img1, img2):
    """
    计算MSE (Mean Squared Error)

    Args:
        img1: 图片1
        img2: 图片2

    Returns:
        mse_value: MSE值
    """
    if isinstance(img1, torch.Tensor):
        return torch.mean((img1 - img2) ** 2).item()
    else:
        return np.mean((img1 - img2) ** 2)


def evaluate_harmonization(pred, target, mask=None):
    """
    综合评估图像协调效果

    Args:
        pred: 预测图像 (torch tensor, range [-1, 1])
        target: 目标图像 (torch tensor, range [-1, 1])
        mask: 掩码（可选，如果提供则计算mask区域的指标）

    Returns:
        metrics: 字典，包含各种指标
    """
    # 反归一化到[0, 1]
    pred_norm = (pred + 1) / 2
    target_norm = (target + 1) / 2

    metrics = {}

    # 全图指标
    metrics['psnr'] = calculate_psnr(pred_norm, target_norm, data_range=1.0)
    metrics['ssim'] = calculate_ssim(pred_norm, target_norm, data_range=1.0)
    metrics['mae'] = calculate_mae(pred_norm, target_norm)
    metrics['mse'] = calculate_mse(pred_norm, target_norm)

    # 如果提供了mask，计算mask区域指标
    if mask is not None:
        mask_binary = (mask > 0.5).float()
        pred_masked = pred_norm * mask_binary
        target_masked = target_norm * mask_binary

        metrics['psnr_mask'] = calculate_psnr(pred_masked, target_masked, data_range=1.0)
        metrics['mae_mask'] = calculate_mae(pred_masked, target_masked)

    return metrics


if __name__ == '__main__':
    # 测试
    img1 = torch.randn(2, 3, 256, 256)
    img2 = torch.randn(2, 3, 256, 256)
    mask = torch.ones(2, 1, 256, 256)

    metrics = evaluate_harmonization(img1, img2, mask)

    print("评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
