"""
模型模块 - 包含三种不同的图像协调模型

方案1: U-Net从头训练 (pctnet.py)
方案2: Libcom预训练PCTNet (libcom_pctnet_wrapper.py)
方案3: ResNet-UNet with ImageNet预训练 (resnet_unet.py)
"""

# 方案1: U-Net从头训练
from .pctnet import PCTNet, create_pctnet, load_pretrained_pctnet

# 方案2: Libcom PCTNet包装器
try:
    from .libcom_pctnet_wrapper import (
        LibcomPCTNetWrapper,
        SimplifiedLibcomWrapper,
        create_libcom_pctnet
    )
    LIBCOM_AVAILABLE = True
except ImportError:
    LIBCOM_AVAILABLE = False
    print("注意: libcom未安装，方案2不可用")

# 方案3: ResNet-UNet
from .resnet_unet import ResNetUNet, create_resnet_unet


__all__ = [
    # 方案1
    'PCTNet',
    'create_pctnet',
    'load_pretrained_pctnet',

    # 方案2 (需要libcom)
    'LibcomPCTNetWrapper',
    'SimplifiedLibcomWrapper',
    'create_libcom_pctnet',
    'LIBCOM_AVAILABLE',

    # 方案3
    'ResNetUNet',
    'create_resnet_unet',
]


def get_model(model_type='unet', **kwargs):
    """
    统一的模型创建接口

    Args:
        model_type: 模型类型
            - 'unet': U-Net从头训练 (方案1)
            - 'libcom': Libcom预训练PCTNet (方案2)
            - 'resnet': ResNet-UNet (方案3)
        **kwargs: 传递给模型创建函数的参数

    Returns:
        model: 创建的模型
    """
    if model_type == 'unet':
        return create_pctnet(**kwargs)
    elif model_type == 'libcom':
        if not LIBCOM_AVAILABLE:
            raise ImportError("libcom未安装，无法使用方案2。请运行: pip install libcom")
        return create_libcom_pctnet(**kwargs)
    elif model_type == 'resnet':
        return create_resnet_unet(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}. 可选: 'unet', 'libcom', 'resnet'")
