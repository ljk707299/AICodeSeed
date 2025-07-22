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

# 随机初始化权重参数
torch.manual_seed(1)
w1 = torch.randn(input_dim, hidden_dim, device=device, dtype=dtype)
w2 = torch.randn(hidden_dim, output_dim, device=device, dtype=dtype)

learning_rate = 1e-6
for epoch in range(500):
    # 前向传播：输入x经过第一层线性变换和ReLU激活，再经过第二层线性变换得到预测输出
    h = x.mm(w1)  # 第一层线性变换
    h_relu = h.clamp(min=0)  # ReLU激活函数
    y_pred = h_relu.mm(w2)  # 第二层线性变换，得到预测值

    # 计算损失函数（均方误差）
    loss = (y_pred - y).pow(2).sum().item()
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # 反向传播：计算损失对各参数的梯度
    grad_y_pred = 2.0 * (y_pred - y)  # 损失对y_pred的梯度
    grad_w2 = h_relu.t().mm(grad_y_pred)  # 损失对w2的梯度
    grad_h_relu = grad_y_pred.mm(w2.t())  # 损失对h_relu的梯度
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0  # ReLU对负值的梯度为0
    grad_w1 = x.t().mm(grad_h)  # 损失对w1的梯度

    # 更新参数（梯度下降）
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2