import torch

# 自动选择设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# 批量大小、输入维度、隐藏层维度、输出维度
batch_size = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

# 随机生成输入和输出数据，并移动到设备上
torch.manual_seed(0)
x = torch.randn(batch_size, input_dim, device=device)
y = torch.randn(batch_size, output_dim, device=device)

# 使用nn.Sequential定义三层神经网络，包含ReLU激活
model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dim)
).to(device)

# 定义均方误差损失函数
loss_fn = torch.nn.MSELoss()

learning_rate = 1e-4
for epoch in range(500):
    # 前向传播：输入x经过模型得到预测输出
    y_pred = model(x)

    # 计算损失函数
    loss = loss_fn(y_pred, y)
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 梯度清零
    model.zero_grad()

    # 反向传播，计算梯度
    loss.backward()

    # 更新参数（梯度下降）
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
