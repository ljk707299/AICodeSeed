import torch

# 创建一个可求导的标量张量x
x = torch.tensor(2.0, requires_grad=True)
# 构建一个简单的线性表达式y = 2x + 3
y = 2 * x + 3
# 对y关于x求导
y.backward()
# 输出x的梯度（即dy/dx=2）
print(f"x的梯度: {x.grad}")