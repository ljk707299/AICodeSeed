#张量的基本操作

import torch
# ==============================
# 1. 张量的基本操作
# ==============================
print("="*50)
print("1. 张量基本操作示例")
print("="*50)

# 创建张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
y = torch.randn(2, 3)  # 正态分布随机张量
z = torch.zeros(3, 2)
print(f"创建张量:\n x={x}\n y={y}\n z={z}")

# 索引和切片
print("\n索引和切片:")
print("x[1, 2] =", x[1, 2].item())  # 获取标量值
print("x[:, 1:] =\n", x[:, 1:])

# 形状变换
reshaped = x.view(3, 2)  # 视图操作(不复制数据)
transposed = x.t()       # 转置
squeezed = torch.randn(1, 3, 1).squeeze()  # 压缩维度
print(f"\n形状变换:\n 重塑后: {reshaped.shape}\n 转置后: {transposed.shape}\n 压缩后: {squeezed.shape}")

# 数学运算
add = x + y              # 逐元素加法
matmul = x @ transposed  # 矩阵乘法
sum_x = x.sum(dim=1)     # 沿维度求和
print(f"\n数学运算:\n 加法:\n{add}\n 矩阵乘法:\n{matmul}\n 行和: {sum_x}")

# 广播机制
a = torch.tensor([1, 2, 3])
b = torch.tensor([[10], [20]])
print(a.shape)
print(b.shape)
print(f"\n广播加法:\n{a + b}")

# 内存共享验证
view_tensor = x.view(6)
view_tensor[0] = 100
print("\n内存共享验证(修改视图影响原始张量):")
print(f"视图: {view_tensor}\n原始: {x}")

