"""
测试修复后的libcom包装器的forward方法
"""
import torch
import sys
sys.path.append('.')
from models.libcom_pctnet_wrapper import create_libcom_pctnet

print("=" * 70)
print("测试 libcom 包装器的 forward 方法")
print("=" * 70)

device = torch.device('cuda')

# 创建模型
print("\n创建模型...")
model = create_libcom_pctnet(device=device)

# 创建测试输入
print("\n创建测试输入...")
batch_size = 2
composite = torch.randn(batch_size, 3, 256, 256).to(device)  # [-1, 1]
mask = torch.ones(batch_size, 1, 256, 256).to(device)  # [0, 1]

print(f"  Composite: {composite.shape}, range [{composite.min():.3f}, {composite.max():.3f}]")
print(f"  Mask: {mask.shape}, range [{mask.min():.3f}, {mask.max():.3f}]")

# 前向传播
print("\n执行前向传播...")
model.eval()
with torch.no_grad():
    output = model(composite, mask)

print(f"\n✓ Forward成功!")
print(f"  输出形状: {output.shape}")
print(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")
print(f"  输出类型: {type(output)}")

# 测试梯度
print("\n测试训练模式（带梯度）...")
model.train()
composite.requires_grad_(True)
output = model(composite, mask)

print(f"  输出需要梯度: {output.requires_grad}")

# 测试反向传播
print("\n测试反向传播...")
loss = output.mean()
loss.backward()

print(f"✓ 反向传播成功!")
print(f"  输入梯度形状: {composite.grad.shape if composite.grad is not None else 'None'}")

print("\n" + "=" * 70)
print("✓ 所有测试通过！libcom包装器可以正常训练")
print("=" * 70)
