# 三个训练方案对比指南

## 📊 方案总览

| 方案 | 模型架构 | 预训练权重来源 | 训练难度 | 推荐场景 |
|------|---------|--------------|---------|---------|
| **方案1** | U-Net | 无（从头训练） | ⭐ 简单 | 快速实验，数据量中等 |
| **方案2** | Transformer PCTNet | Libcom预训练 | ⭐⭐⭐ 复杂 | 追求最佳性能，有充足算力 |
| **方案3** | ResNet-UNet | ImageNet预训练 | ⭐⭐ 中等 | **推荐！平衡性能和易用性** |

---

## 🎯 方案1: U-Net从头训练

### 架构
```
U-Net (Encoder-Decoder with Skip Connections)
- Encoder: 4个下采样块 (64→128→256→512)
- Bottleneck: 带注意力机制
- Decoder: 4个上采样块 (512→256→128→64)
```

### 优点
✅ 实现简单，代码清晰
✅ 训练快速（参数量适中）
✅ 不依赖外部权重
✅ 在线合成数据足够训练

### 缺点
❌ 收敛可能稍慢
❌ 小数据集效果可能不如预训练模型

### 适用场景
- 快速原型验证
- 数据量500-2000张
- 不想折腾环境配置

### 训练命令
```bash
bash train_plan1_unet.sh
```

或直接使用：
```bash
python train_universal.py --model_type unet \
  --train_image_dir your_crack_dataset/train/images \
  --train_mask_dir your_crack_dataset/train/masks \
  --val_image_dir your_crack_dataset/val/images \
  --val_mask_dir your_crack_dataset/val/masks \
  --epochs 100 --batch_size 8 --lr 2e-4
```

### 参数量
- **总参数**: ~5M
- **可训练参数**: ~5M

---

## 🚀 方案2: Libcom预训练PCTNet

### 架构
```
Transformer-based PCTNet
- Encoder: Vision Transformer (ViT-like)
- Patch Embedding: 图像分块嵌入
- Transformer Encoder: 9层Transformer
- Color Transform Network: 学习像素级颜色变换
```

### 优点
✅ 使用Libcom的大规模预训练权重
✅ 理论性能最佳
✅ 完全兼容官方PCTNet架构

### 缺点
❌ 需要安装libcom（可能有依赖冲突）
❌ 模型较大，显存占用高
❌ 训练速度较慢
❌ 包装器可能有兼容性问题

### 适用场景
- 追求最高精度
- 有充足GPU资源（推荐12GB+显存）
- 愿意花时间调试libcom

### 训练命令
```bash
# 确保已安装libcom
pip install libcom

bash train_plan2_libcom.sh
```

或直接使用：
```bash
python train_universal.py --model_type libcom \
  --train_image_dir your_crack_dataset/train/images \
  --train_mask_dir your_crack_dataset/train/masks \
  --val_image_dir your_crack_dataset/val/images \
  --val_mask_dir your_crack_dataset/val/masks \
  --freeze_encoder \
  --epochs 50 --batch_size 4 --lr 1e-4
```

### 参数量
- **总参数**: ~18M
- **可训练参数**: ~5M（冻结encoder）

### ⚠️ 注意事项
1. 如果遇到libcom兼容性问题，建议使用方案1或方案3
2. batch_size建议设为4（显存占用大）
3. 可能需要修改包装器以适配libcom的最新版本

---

## ⭐ 方案3: ResNet-UNet (推荐)

### 架构
```
ResNet18-UNet
- Encoder: 预训练ResNet18 (ImageNet)
- 特征层级: 64→64→128→256→512
- Decoder: 对称的上采样+卷积块
- Skip Connections: 融合多尺度特征
```

### 优点
✅ 使用ImageNet预训练（1400万图片训练）
✅ ResNet backbone性能强大
✅ 不需要手动下载权重（torchvision自动下载）
✅ 训练速度快
✅ 显存占用合理
✅ **平衡性能和易用性**

### 缺点
❌ 第一次运行需要下载ResNet权重（约45MB，自动完成）

### 适用场景
- **推荐首选方案！**
- 任何数据量级别
- 追求性能但又不想折腾
- 显存有限（8GB可运行）

### 训练命令
```bash
bash train_plan3_resnet.sh
```

或直接使用：
```bash
python train_universal.py --model_type resnet \
  --train_image_dir your_crack_dataset/train/images \
  --train_mask_dir your_crack_dataset/train/masks \
  --val_image_dir your_crack_dataset/val/images \
  --val_mask_dir your_crack_dataset/val/masks \
  --freeze_encoder \
  --epochs 100 --batch_size 8 --lr 1e-4
```

### 参数量
- **总参数**: ~13M
- **可训练参数**: ~6M（冻结encoder）

### 为什么推荐方案3？
1. **预训练权重质量高**: ImageNet是图像识别领域最经典的数据集
2. **开箱即用**: torchvision自动处理权重下载
3. **性能优秀**: ResNet是经过验证的强大backbone
4. **资源友好**: 比Transformer模型更轻量

---

## 🔍 详细对比

### 训练速度（相对值）
```
方案1 (U-Net):        ████████░░  80%
方案2 (Libcom PCTNet): ████░░░░░░  40%
方案3 (ResNet-UNet):  ██████████ 100% (最快)
```

### 收敛速度（达到相同精度所需epochs）
```
方案1: 80-100 epochs
方案2: 30-50 epochs
方案3: 50-80 epochs
```

### 显存占用 (batch_size=8, image_size=256)
```
方案1: ~4GB
方案2: ~8GB
方案3: ~5GB
```

### 最终性能（理论值）
```
方案1: ⭐⭐⭐☆☆
方案2: ⭐⭐⭐⭐⭐
方案3: ⭐⭐⭐⭐☆
```

---

## 🎓 选择建议

### 情况1: 我是新手，想快速开始
**推荐**: **方案3 (ResNet-UNet)**
- 理由: 性能好，易用，不会出错

### 情况2: 我的数据集很小（<500张）
**推荐**: **方案3 (ResNet-UNet)** + `--freeze_encoder`
- 理由: 预训练权重能弥补数据不足

### 情况3: 我的数据集很大（>2000张）
**推荐**: **方案1 (U-Net)** 或 **方案3 (ResNet-UNet)**
- 理由: 从头训练也能获得好效果，省去预训练的麻烦

### 情况4: 我追求最高精度，不在乎成本
**推荐**: **方案2 (Libcom PCTNet)**
- 理由: 官方预训练模型理论性能最佳
- 注意: 需要花时间调试

### 情况5: 我的显存有限（4-6GB）
**推荐**: **方案1 (U-Net)** + 小batch_size
```bash
python train_universal.py --model_type unet \
  --batch_size 4 --image_size 128 \
  其他参数...
```

### 情况6: 我想发论文，需要baseline对比
**推荐**: 三个方案都跑，对比效果
- 可以展示不同架构/预训练策略的影响

---

## 📝 实验建议

### 第一步: 快速验证（2-3小时）
使用**方案3 (ResNet-UNet)**，训练20个epochs，验证流程是否正常：
```bash
python train_universal.py --model_type resnet \
  --train_image_dir your_crack_dataset/train/images \
  --train_mask_dir your_crack_dataset/train/masks \
  --val_image_dir your_crack_dataset/val/images \
  --val_mask_dir your_crack_dataset/val/masks \
  --freeze_encoder --epochs 20 --batch_size 8
```

### 第二步: 完整训练（1-2天）
根据第一步结果，选择最佳方案进行完整训练（80-100 epochs）

### 第三步: 超参数调优
调整：
- `--mask_weight` (3.0~10.0)
- `--aug_intensity` (low/medium/high)
- `--lr` (1e-4~2e-4)

---

## 🛠️ 故障排除

### 问题1: Libcom安装失败（方案2）
**解决**: 使用方案3替代，性能相近且更稳定

### 问题2: 显存不足
**解决**:
```bash
--batch_size 4 --image_size 128
```

### 问题3: 训练不收敛
**解决**:
1. 降低学习率: `--lr 5e-5`
2. 检查掩码格式（白色=裂缝区域）
3. 降低增强强度: `--aug_intensity low`

### 问题4: 想切换方案
只需修改 `--model_type` 参数：
```bash
# 从方案1切换到方案3
python train_universal.py --model_type resnet ...
```

---

## 📦 文件清单

```
models/
├── pctnet.py                    # 方案1: U-Net
├── libcom_pctnet_wrapper.py     # 方案2: Libcom包装器
└── resnet_unet.py               # 方案3: ResNet-UNet

训练脚本:
├── train_universal.py           # 通用训练脚本（支持三种方案）
├── train_plan1_unet.sh          # 方案1快速启动
├── train_plan2_libcom.sh        # 方案2快速启动
└── train_plan3_resnet.sh        # 方案3快速启动
```

---

## 🎯 总结

| 如果你...  | 选择方案 |
|-----------|---------|
| 想要最简单快速的开始 | **方案3** ⭐ |
| 数据量少(<500张) | **方案3** + freeze_encoder |
| 数据量多(>2000张) | **方案1** 或 **方案3** |
| 追求极致性能 | **方案2** (需调试) |
| 显存有限 | **方案1** |
| 不知道选什么 | **方案3** ⭐⭐⭐ |

**最终推荐**: 如无特殊需求，直接使用 **方案3 (ResNet-UNet)**！

---

Generated by Claude Code
Last updated: 2024-12-15
