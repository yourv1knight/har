"""
道路裂缝图像协调训练脚本
支持:
- 从libcom预训练权重开始微调
- MSE Loss + 对比损失
- Tensorboard可视化
- 定期保存checkpoints
"""
import os
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 项目内导入
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.pctnet import create_pctnet
from datasets.crack_dataset import CrackHarmonizationDataset


class ReconstructionLoss(nn.Module):
    """重建损失 = MSE Loss (在全图) + MSE Loss (在Mask区域，加权)"""

    def __init__(self, mask_weight=5.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mask_weight = mask_weight

    def forward(self, pred, target, mask):
        # 全图损失
        loss_global = self.mse(pred, target)

        # Mask区域损失（重点关注）
        mask_binary = (mask > 0.5).float()
        loss_mask = self.mse(pred * mask_binary, target * mask_binary)

        # 总损失
        total_loss = loss_global + self.mask_weight * loss_mask

        return total_loss, loss_global, loss_mask


class PerceptualLoss(nn.Module):
    """
    简化的感知损失（可选）
    使用预训练VGG提取特征，计算特征空间距离
    """

    def __init__(self, device='cuda'):
        super().__init__()
        # 这里简化实现，如果需要可以集成torchvision的VGG
        # 为了减少依赖，暂时使用MSE替代
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 简化版：直接使用像素级MSE
        # 真实项目中应该使用VGG特征
        return self.mse(pred, target)


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, global_step):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_loss_global = 0.0
    epoch_loss_mask = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # 数据移到GPU
        composite = batch['composite'].to(device)
        real = batch['real'].to(device)
        mask = batch['mask'].to(device)

        # 前向传播
        optimizer.zero_grad()
        pred = model(composite, mask)

        # 计算损失
        loss, loss_global, loss_mask = criterion(pred, real, mask)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录
        epoch_loss += loss.item()
        epoch_loss_global += loss_global.item()
        epoch_loss_mask += loss_mask.item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'global': f'{loss_global.item():.4f}',
            'mask': f'{loss_mask.item():.4f}'
        })

        # Tensorboard记录（每N步）
        if global_step % 50 == 0:
            writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
            writer.add_scalar('Train/Loss_Global', loss_global.item(), global_step)
            writer.add_scalar('Train/Loss_Mask', loss_mask.item(), global_step)

        global_step += 1

    # Epoch统计
    avg_loss = epoch_loss / len(dataloader)
    avg_loss_global = epoch_loss_global / len(dataloader)
    avg_loss_mask = epoch_loss_mask / len(dataloader)

    return avg_loss, avg_loss_global, avg_loss_mask, global_step


def validate(model, dataloader, criterion, device, epoch, writer):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    val_loss_global = 0.0
    val_loss_mask = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            composite = batch['composite'].to(device)
            real = batch['real'].to(device)
            mask = batch['mask'].to(device)

            # 前向传播
            pred = model(composite, mask)

            # 计算损失
            loss, loss_global, loss_mask = criterion(pred, real, mask)

            val_loss += loss.item()
            val_loss_global += loss_global.item()
            val_loss_mask += loss_mask.item()

            # 可视化前几张图片
            if batch_idx == 0 and writer is not None:
                # 反归一化到[0,1]
                composite_vis = (composite[:4] + 1) / 2
                real_vis = (real[:4] + 1) / 2
                pred_vis = (pred[:4] + 1) / 2
                mask_vis = mask[:4].repeat(1, 3, 1, 1)  # 转为3通道便于可视化

                # 拼接图片
                comparison = torch.cat([composite_vis, real_vis, pred_vis, mask_vis], dim=0)
                writer.add_images('Val/Comparison', comparison, epoch)

    avg_loss = val_loss / len(dataloader)
    avg_loss_global = val_loss_global / len(dataloader)
    avg_loss_mask = val_loss_mask / len(dataloader)

    # Tensorboard记录
    if writer is not None:
        writer.add_scalar('Val/Loss_Total', avg_loss, epoch)
        writer.add_scalar('Val/Loss_Global', avg_loss_global, epoch)
        writer.add_scalar('Val/Loss_Mask', avg_loss_mask, epoch)

    return avg_loss, avg_loss_global, avg_loss_mask


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.exp_name}_{timestamp}"
    exp_dir = Path(args.output_dir) / exp_name

    checkpoint_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Log dir: {log_dir}")

    # Tensorboard
    writer = SummaryWriter(log_dir=str(log_dir))

    # 1. 创建数据集
    print("\n[1/5] Loading dataset...")
    train_dataset = CrackHarmonizationDataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        image_size=args.image_size,
        mode='train',
        augmentation_intensity=args.aug_intensity
    )

    val_dataset = CrackHarmonizationDataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        image_size=args.image_size,
        mode='val',
        augmentation_intensity='low'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # 2. 创建模型
    print("\n[2/5] Creating model...")
    model = create_pctnet(
        pretrained_path=args.pretrained_path,
        freeze_encoder=args.freeze_encoder,
        device=device
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # 3. 损失函数和优化器
    print("\n[3/5] Setting up training...")
    criterion = ReconstructionLoss(mask_weight=args.mask_weight)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.lr_decay_gamma
    )

    # 4. 恢复训练（如果有checkpoint）
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume:
        if os.path.exists(args.resume):
            print(f"  Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['loss']
            print(f"  Resumed from epoch {start_epoch}")

    # 5. 训练循环
    print("\n[4/5] Start training...")
    print("=" * 80)

    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # 训练
        train_loss, train_global, train_mask, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer, global_step
        )

        print(f"Train Loss: {train_loss:.4f} (Global: {train_global:.4f}, Mask: {train_mask:.4f})")

        # 验证
        val_loss, val_global, val_mask = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        print(f"Val Loss:   {val_loss:.4f} (Global: {val_global:.4f}, Mask: {val_mask:.4f})")

        # 学习率衰减
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        print(f"Learning Rate: {current_lr:.6f}")

        # 保存checkpoint
        if epoch % args.save_interval == 0:
            save_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)
            print(f"✓ New best model! Val Loss: {val_loss:.4f}")

    # 保存最终模型
    final_path = checkpoint_dir / "final_model.pth"
    save_checkpoint(model, optimizer, args.epochs, val_loss, final_path)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Tensorboard logs: {log_dir}")
    print("\nTo view training logs, run:")
    print(f"  tensorboard --logdir {log_dir.parent}")
    print("=" * 80)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train crack harmonization model')

    # 数据路径
    parser.add_argument('--train_image_dir', type=str, required=False,default='your_crack_dataset/train/images', 
                        help='Training image directory')
    parser.add_argument('--train_mask_dir', type=str, required=False,default='your_crack_dataset/train/masks', 
                        help='Training mask directory')
    parser.add_argument('--val_image_dir', type=str, required=False,default='your_crack_dataset/val/images', 
                        help='Validation image directory')
    parser.add_argument('--val_mask_dir', type=str, required=False,default='your_crack_dataset/val/masks', 
                        help='Validation mask directory')

    # 模型配置
    parser.add_argument('--pretrained_path', type=str, default='/home/enine/anaconda3/envs/DIH/lib/python3.10/site-packages/libcom/image_harmonization/pretrained_models/PCTNet.pth',
                        help='Path to pretrained model weights (from libcom)')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder layers for fine-tuning')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lr_decay_step', type=int, default=30,
                        help='LR decay step size (epochs)')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5,
                        help='LR decay gamma')

    # 损失函数
    parser.add_argument('--mask_weight', type=float, default=5.0,
                        help='Weight for mask region loss')

    # 数据增强
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--aug_intensity', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Data augmentation intensity')

    # 其他
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='./crack_harmonization_train/experiments',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--exp_name', type=str, default='pctnet_crack',
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # 打印配置
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:25s}: {value}")
    print("=" * 80)

    main(args)
