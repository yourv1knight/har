"""
配置文件 - 道路裂缝图像协调项目
包含所有路径和模型参数的配置
"""
import os

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
BACKGROUND_DIR = os.path.join(DATA_DIR, 'backgrounds')  # 背景图片（干净路面）
FOREGROUND_DIR = os.path.join(DATA_DIR, 'foregrounds')  # 前景图片（裂缝）
MASK_DIR = os.path.join(DATA_DIR, 'masks')              # 裂缝掩码

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# 模型权重路径
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# ==================== 模型配置 ====================
# libcom支持的协调模型
# 可选: 'doveNet', 'rainNet', 'htNet', 'dccNet', 'bctNet', 'pctNet', 'iscNet'
# 推荐使用 'pctNet'（PCT-Net）或 'rainNet'（RainNet），效果较好
HARMONIZATION_MODEL = 'pctNet'

# 设备配置（自动检测）
DEVICE = 'cuda'  # 如果没有GPU，会自动切换到CPU

# ==================== 合成参数 ====================
# 随机粘贴位置的边界（避免裂缝被裁剪）
PASTE_MARGIN = 50  # 距离图像边缘的最小距离（像素）

# 输出图像质量
OUTPUT_QUALITY = 95  # JPEG质量（1-100）

# ==================== 可视化配置 ====================
# 是否显示结果图像
SHOW_RESULTS = True

# 是否保存对比图
SAVE_COMPARISON = True

# 图像尺寸配置（如果需要调整大小）
MAX_IMAGE_SIZE = None  # 设置为None表示不调整，或设置如 (1024, 1024)

# ==================== 日志配置 ====================
VERBOSE = True  # 是否输出详细信息
