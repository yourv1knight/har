# 道路裂缝图像协调训练项目

完整的 PyTorch 训练工程，用于在道路裂缝数据集上微调 PCTNet 图像协调模型。

## 🎯 项目特点

- ✅ **在线合成（Online Compositing）**：从真实图片实时生成不协调的训练样本
- ✅ **预训练权重加载**：支持加载 libcom 的 PCTNet 预训练权重
- ✅ **灵活微调**：支持冻结 Encoder 层，只训练 Decoder
- ✅ **可视化训练**：集成 Tensorboard 实时监控
- ✅ **完整推理**：包含单张/批量推理脚本

---

## 📁 项目结构

```
crack_harmonization_train/
├── models/                     # 模型定义
│   ├── __init__.py
│   └── pctnet.py              # PCTNet实现（带预训练加载）
├── datasets/                   # 数据集
│   ├── __init__.py
│   └── crack_dataset.py       # 在线合成数据集
├── train.py                   # 训练脚本
├── inference_check.py         # 推理验证脚本
├── requirements.txt           # 依赖包
├── configs/                   # 配置文件目录（可选）
├── checkpoints/               # 模型权重保存目录
├── logs/                      # Tensorboard日志
└── README.md                  # 本文件
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（推荐）
conda create -n crack_harmonization python=3.9
conda activate crack_harmonization

# 安装依赖
cd crack_harmonization_train
pip install -r requirements.txt
```

### 2. 数据准备

准备你的道路裂缝数据集，组织成以下结构：

```
your_crack_dataset/
├── train/
│   ├── images/           # 真实裂缝图片
│   │   ├── crack_001.jpg
│   │   ├── crack_002.jpg
│   │   └── ...
│   └── masks/            # 对应的掩码（白色=裂缝区域）
│       ├── crack_001.jpg (或 crack_001_mask.png)
│       ├── crack_002.jpg
│       └── ...
└── val/
    ├── images/
    └── masks/
```

**重要提示**：
- 图片和掩码文件名需要能够匹配（完全相同或有 `_mask` 后缀）
- 掩码格式：白色（255）= 裂缝区域，黑色（0）= 背景

### 3. 训练模型

#### 3.1 从头训练（不使用预训练权重）

```bash
python train.py \
  --train_image_dir /path/to/your_crack_dataset/train/images \
  --train_mask_dir /path/to/your_crack_dataset/train/masks \
  --val_image_dir /path/to/your_crack_dataset/val/images \
  --val_mask_dir /path/to/your_crack_dataset/val/masks \
  --epochs 100 \
  --batch_size 8 \
  --lr 2e-4 \
  --image_size 256 \
  --exp_name pctnet_crack_baseline
```

#### 3.2 从 libcom 预训练权重微调（推荐）

首先，找到 libcom 下载的 PCTNet 权重文件（通常在 `~/.cache/libcom/` 或你的 checkpoints 目录）。

```bash
# 示例：使用预训练权重 + 冻结Encoder
python train.py \
  --train_image_dir /path/to/your_crack_dataset/train/images \
  --train_mask_dir /path/to/your_crack_dataset/train/masks \
  --val_image_dir /path/to/your_crack_dataset/val/images \
  --val_mask_dir /path/to/your_crack_dataset/val/masks \
  --pretrained_path /path/to/libcom_pctnet_weights.pth \
  --freeze_encoder \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --exp_name pctnet_crack_finetune
```

**微调技巧**：
- `--freeze_encoder`：冻结编码器，只训练解码器（推荐用于小数据集）
- 微调时使用较小的学习率（1e-4）和较少的 epochs（50-100）

#### 3.3 恢复训练

```bash
python train.py \
  --resume ./experiments/pctnet_crack_20241215_120000/checkpoints/checkpoint_epoch_0030.pth \
  --train_image_dir ... \
  --train_mask_dir ... \
  --val_image_dir ... \
  --val_mask_dir ...
```

---

## 🔍 监控训练

启动 Tensorboard 查看训练曲线：

```bash
tensorboard --logdir ./experiments
```

然后在浏览器打开 `http://localhost:6006`

你可以看到：
- **Train/Loss_Total**：总损失
- **Train/Loss_Global**：全图MSE损失
- **Train/Loss_Mask**：裂缝区域MSE损失
- **Val/Comparison**：验证集图片对比

---

## 🧪 推理验证

### 单张图片推理

```bash
python inference_check.py \
  --checkpoint ./experiments/pctnet_crack_*/checkpoints/best_model.pth \
  --image /path/to/test_crack.jpg \
  --mask /path/to/test_crack_mask.jpg \
  --output result_harmonized.jpg
```

### 批量推理

```bash
python inference_check.py \
  --checkpoint ./experiments/pctnet_crack_*/checkpoints/best_model.pth \
  --batch \
  --image_dir /path/to/test_images \
  --mask_dir /path/to/test_masks \
  --output_dir ./inference_results
```

---

## ⚙️ 核心参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 8 | 批量大小（根据GPU显存调整） |
| `--lr` | 2e-4 | 学习率（微调建议 1e-4） |
| `--image_size` | 256 | 训练图像尺寸 |
| `--mask_weight` | 5.0 | 裂缝区域损失权重 |
| `--freeze_encoder` | False | 是否冻结编码器 |
| `--aug_intensity` | medium | 数据增强强度（low/medium/high） |

### 数据增强强度

在 `datasets/crack_dataset.py` 中，通过 LUT 和 HSV 变换制造"不协调感"：

- **low**：轻微颜色偏移（适合验证集）
- **medium**：中等强度（默认，推荐训练）
- **high**：强烈颜色变化（适合数据量极少时）

---

## 🧠 核心设计：在线合成逻辑

### 为什么需要在线合成？

传统的图像协调数据集需要：
1. 前景图片（裂缝）
2. 背景图片（路面）
3. 手动合成得到"不协调的合成图"
4. 使用真实图作为Ground Truth

但在垂直领域（如道路裂缝），我们只有：
- 真实的裂缝图片
- 对应的裂缝掩码

### 我们的解决方案：逆向数据生成

```
真实图片 (Real)
    ↓
在Mask区域应用数据增强（LUT/HSV变换）
    ↓
生成不协调的合成图 (Composite)
    ↓
[模型训练] Composite → Real
```

**关键代码**（在 `datasets/crack_dataset.py`）：

```python
def _generate_disharmony(self, real_image, mask_image):
    """
    核心：仅在Mask区域内改变颜色/亮度，制造不协调感
    """
    # 方法1: LUT变换（70%概率）
    # 方法2: HSV变换（30%概率）
    # 可选: 添加轻微噪声
```

---

## 📊 预期效果

### 训练收敛情况

- **前10个epoch**：Loss快速下降
- **20-50 epoch**：逐渐收敛
- **最佳验证Loss**：通常在0.01-0.05之间（取决于数据质量）

### 如何判断效果？

1. **观察 Tensorboard 中的 Val/Comparison**
   - Composite（输入）应该看起来不协调
   - Harmonized（输出）应该更自然

2. **使用 inference_check.py 推理测试集**
   - 对比原图和协调后的图像
   - 裂缝区域应该与背景更融合

---

## 🛠️ 常见问题

### Q1: 找不到 libcom 的预训练权重在哪？

运行你的 `run_harmonization.py` 后，libcom 会自动下载权重到：
- Linux: `~/.cache/libcom/`
- 或者检查你的 `checkpoints/` 目录

找到类似 `pctnet_xxx.pth` 的文件即可。

### Q2: 显存不足怎么办？

```bash
# 方法1: 减小batch_size
--batch_size 4

# 方法2: 减小图像尺寸
--image_size 128

# 方法3: 冻结Encoder（减少梯度计算）
--freeze_encoder
```

### Q3: 数据增强太强/太弱？

修改 `--aug_intensity` 参数，或直接编辑 `datasets/crack_dataset.py` 中的 `_get_augmentation_params()` 方法。

### Q4: 训练不收敛？

可能的原因：
1. **学习率过大**：尝试 `--lr 1e-4` 或 `5e-5`
2. **数据质量问题**：检查掩码是否正确（白色=裂缝）
3. **增强过强**：降低 `--aug_intensity` 到 `low`

### Q5: 如何调用训练好的模型？

```python
from models.pctnet import PCTNet
import torch

# 加载模型
model = PCTNet()
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    output = model(composite_image, mask)
```

---

## 📚 引用

如果使用 PCTNet，请引用原论文：

```
@inproceedings{pctnet,
  title={PCT-Net: Full Resolution Image Harmonization Using Pixel-Wise Color Transforms},
  author={...},
  booktitle={CVPR},
  year={2023}
}
```

---

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

## 🎉 下一步

1. **数据准备**：整理你的裂缝数据集
2. **测试数据集**：运行 `python datasets/crack_dataset.py` 查看数据增强效果
3. **开始训练**：使用上述命令开始微调
4. **评估效果**：使用 `inference_check.py` 测试真实场景

祝训练顺利！🚀
