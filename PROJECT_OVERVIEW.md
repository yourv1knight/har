# 项目总览 - 道路裂缝图像协调训练工程

## 📋 项目信息

**项目名称**: 道路裂缝图像协调训练系统
**目标**: 在垂直领域（道路裂缝）数据集上微调 PCTNet 图像协调模型
**创建日期**: 2024-12-15

---

## 🎯 核心创新点

### 1. 在线合成逻辑（Online Compositing）
传统方法需要预先准备"不协调的合成图"，但本项目实现了**逆向数据生成**：

```
真实裂缝图 + 掩码
    ↓
在Mask区域应用颜色/亮度增强（LUT/HSV变换）
    ↓
生成"假的不协调合成图"
    ↓
训练模型: 不协调图 → 真实图
```

**优势**：
- ✅ 不需要额外的背景图片
- ✅ 数据利用率高（一张图可生成多个变体）
- ✅ 更贴近真实场景（直接从真实裂缝学习）

### 2. 支持预训练权重微调
- 可以加载 libcom 的 PCTNet 预训练权重
- 支持冻结 Encoder，只训练 Decoder
- 适合小数据集场景

### 3. 完整的训练/推理流程
- Tensorboard 可视化
- 自动保存最佳模型
- 批量推理支持

---

## 📂 完整文件结构

```
crack_harmonization_train/
│
├── datasets/                          # 数据集模块
│   ├── __init__.py
│   └── crack_dataset.py              # 核心：在线合成数据集
│       ├── CrackHarmonizationDataset  # 主数据集类
│       ├── _generate_disharmony()     # 生成不协调图（LUT/HSV）
│       ├── _apply_lut_transform()     # LUT颜色变换
│       └── _apply_hsv_transform()     # HSV空间变换
│
├── models/                            # 模型定义
│   ├── __init__.py
│   └── pctnet.py                     # PCTNet网络
│       ├── PCTNet                     # U-Net + Attention架构
│       ├── create_pctnet()            # 创建模型的便捷函数
│       ├── load_pretrained_pctnet()   # 加载预训练权重
│       └── freeze_encoder()           # 冻结Encoder层
│
├── utils/                             # 工具函数
│   ├── __init__.py
│   └── metrics.py                    # 评估指标
│       ├── calculate_psnr()           # PSNR计算
│       ├── calculate_ssim()           # SSIM计算
│       ├── calculate_mae()            # MAE计算
│       └── evaluate_harmonization()   # 综合评估
│
├── train.py                          # 训练脚本
│   ├── ReconstructionLoss            # 重建损失（全图+Mask区域）
│   ├── train_one_epoch()              # 训练一个epoch
│   ├── validate()                     # 验证模型
│   └── save_checkpoint()              # 保存权重
│
├── inference_check.py                # 推理验证脚本
│   ├── load_model()                   # 加载训练好的模型
│   ├── harmonize_image()              # 单张图片推理
│   ├── batch_inference()              # 批量推理
│   └── visualize_results()            # 可视化对比
│
├── examples.py                       # 使用示例代码
│   ├── example_1: 基本推理
│   ├── example_2: 真实图片推理
│   ├── example_3: 训练循环示意
│   └── example_4: 冻结Encoder微调
│
├── configs/                          # 配置文件
│   └── train_config_example.yaml    # 训练配置模板
│
├── checkpoints/                      # 模型权重保存目录
├── logs/                             # Tensorboard日志目录
├── test_data/                        # 测试数据目录
│
├── requirements.txt                  # Python依赖
├── README.md                         # 使用文档
├── PROJECT_OVERVIEW.md               # 本文件
└── quick_test.sh                     # 快速测试脚本

```

---

## 🔧 核心模块详解

### 1. datasets/crack_dataset.py
**核心功能**：在线生成训练样本

```python
class CrackHarmonizationDataset:
    def __getitem__(self, idx):
        # 1. 读取真实图和掩码
        real_image, mask = self._load_pair(idx)

        # 2. 【关键】生成不协调的合成图
        composite_image = self._generate_disharmony(real_image, mask)

        # 3. 返回训练对
        return {
            'composite': composite_image,  # 输入
            'real': real_image,            # Ground Truth
            'mask': mask                   # 协调区域
        }
```

**增强策略**：
- **LUT变换（70%）**：通过查找表改变颜色映射
- **HSV变换（30%）**：在HSV空间随机调整色调/饱和度/亮度
- **可选噪声**：增加真实感

### 2. models/pctnet.py
**架构**：U-Net + Attention

```
输入: [Composite (3ch) + Mask (1ch)] = 4 channels
    ↓
Encoder (可冻结)
    ↓
Bottleneck + Attention
    ↓
Decoder (Skip Connections)
    ↓
输出: Harmonized Image (3 channels)
```

**特性**：
- 支持加载 libcom 预训练权重
- 支持冻结 Encoder（freeze_encoder()）
- 残差连接: `output = input + adjustment`

### 3. train.py
**损失函数**：
```python
Loss_total = MSE_global + α * MSE_mask
```
- `MSE_global`: 全图重建损失
- `MSE_mask`: 裂缝区域重建损失（权重α=5.0）

**训练特性**：
- Adam 优化器（lr=2e-4）
- StepLR 学习率衰减
- Tensorboard 实时可视化
- 自动保存最佳模型

### 4. inference_check.py
**功能**：
- 单张图片推理 + 可视化对比
- 批量处理整个目录
- 支持恢复原始图片尺寸

---

## 🚀 快速使用流程

### 步骤1: 安装环境
```bash
pip install -r requirements.txt
bash quick_test.sh  # 测试环境
```

### 步骤2: 准备数据
```
your_dataset/
├── train/
│   ├── images/  # 真实裂缝图片
│   └── masks/   # 掩码（白色=裂缝）
└── val/
    ├── images/
    └── masks/
```

### 步骤3: 训练
```bash
# 从头训练
python train.py \
  --train_image_dir /path/to/train/images \
  --train_mask_dir /path/to/train/masks \
  --val_image_dir /path/to/val/images \
  --val_mask_dir /path/to/val/masks \
  --epochs 100 --batch_size 8

# 或从预训练权重微调（推荐）
python train.py \
  --pretrained_path /path/to/libcom_pctnet.pth \
  --freeze_encoder \
  --lr 1e-4 --epochs 50 \
  [其他参数同上]
```

### 步骤4: 推理
```bash
# 单张图片
python inference_check.py \
  --checkpoint ./experiments/*/checkpoints/best_model.pth \
  --image test.jpg --mask test_mask.jpg \
  --output result.jpg

# 批量处理
python inference_check.py \
  --checkpoint ./experiments/*/checkpoints/best_model.pth \
  --batch --image_dir ./test_images --mask_dir ./test_masks
```

---

## 📊 性能指标

### 训练监控指标
- **Loss_Total**: 总损失
- **Loss_Global**: 全图MSE
- **Loss_Mask**: 裂缝区域MSE
- **Learning_Rate**: 学习率变化

### 评估指标（utils/metrics.py）
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)

---

## 🎛️ 超参数调优建议

### 关键参数

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `--lr` | 1e-4 ~ 2e-4 | 微调用1e-4，从头训练用2e-4 |
| `--batch_size` | 4 ~ 16 | 取决于GPU显存 |
| `--mask_weight` | 3.0 ~ 10.0 | 裂缝区域损失权重 |
| `--aug_intensity` | medium | low/medium/high |
| `--freeze_encoder` | True | 小数据集推荐冻结 |

### 不同场景的配置

**场景1: 数据量少（<500张）**
```bash
--freeze_encoder --lr 5e-5 --epochs 100 --aug_intensity high
```

**场景2: 数据量中等（500-2000张）**
```bash
--freeze_encoder --lr 1e-4 --epochs 50 --aug_intensity medium
```

**场景3: 数据量大（>2000张）**
```bash
--lr 2e-4 --epochs 100 --aug_intensity medium
```

---

## ⚠️ 常见问题与解决方案

### 问题1: 显存不足
**症状**: `CUDA out of memory`
**解决**:
```bash
--batch_size 4  # 减小batch size
--image_size 128  # 减小图像尺寸
--freeze_encoder  # 冻结Encoder减少梯度计算
```

### 问题2: Loss不下降
**可能原因**:
1. 学习率过大 → 降低到 `--lr 5e-5`
2. 数据质量问题 → 检查掩码是否正确
3. 增强过强 → 降低到 `--aug_intensity low`

### 问题3: 过拟合
**症状**: 训练Loss很低，验证Loss很高
**解决**:
1. 增加数据增强: `--aug_intensity high`
2. 减少训练轮数: `--epochs 30`
3. 使用预训练权重: `--pretrained_path`

### 问题4: 找不到预训练权重
**位置**:
1. 运行 `run_harmonization.py` 后查看 `~/.cache/libcom/`
2. 或手动从 [libcom GitHub](https://github.com/bcmi/libcom) 下载

---

## 📚 技术栈

- **深度学习框架**: PyTorch 2.0+
- **图像处理**: OpenCV, Pillow
- **可视化**: Tensorboard, Matplotlib
- **评估指标**: scikit-image

---

## 🔬 进阶功能（可扩展）

### 1. 添加对比损失（Contrastive Loss）
编辑 `train.py`，在 `ReconstructionLoss` 中添加感知损失或对比损失。

### 2. 多尺度训练
修改 `datasets/crack_dataset.py`，支持随机尺寸训练。

### 3. 集成更多模型
在 `models/` 目录下添加其他协调模型（如 RainNet, DoveNet）。

### 4. 添加数据增强
编辑 `datasets/crack_dataset.py` 的 `_apply_augmentation()` 方法。

---

## 📝 TODO List

- [ ] 添加混合精度训练（AMP）支持
- [ ] 实现分布式训练
- [ ] 添加更多评估指标（FID, LPIPS）
- [ ] 支持从YAML配置文件读取参数
- [ ] 添加模型导出（ONNX）功能

---

## 📧 技术支持

如遇到问题，请检查：
1. README.md - 基础使用文档
2. examples.py - 代码示例
3. 各模块的 docstring 注释

---

**项目作者**: Claude Code
**最后更新**: 2024-12-15
**版本**: v1.0
