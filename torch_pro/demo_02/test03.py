import torch

# 设置数据类型和设备（优先使用GPU）
dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")  # 如果有GPU可用则使用GPU，否则注释掉此行使用CPU

# 批量大小、输入维度、隐藏层维度、输出维度
batch_size = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

# 随机生成输入和输出数据，并放到指定设备
torch.manual_seed(0)
x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
y = torch.randn(batch_size, output_dim, device=device, dtype=dtype)

# 随机初始化权重参数，并设置requires_grad=True以便自动求导
torch.manual_seed(1)
w1 = torch.randn(input_dim, hidden_dim, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(hidden_dim, output_dim, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for epoch in range(500):
    # 前向传播：输入x经过第一层线性变换和ReLU激活，再经过第二层线性变换得到预测输出
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 计算损失函数（均方误差）
    loss = (y_pred - y).pow(2).sum()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 使用autograd自动反向传播，计算梯度
    loss.backward()

    # 手动使用梯度下降更新参数，需在torch.no_grad()上下文中进行
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # 梯度清零，防止累积
        w1.grad.zero_()
        w2.grad.zero_()