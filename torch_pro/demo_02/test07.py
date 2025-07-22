import random
import torch


class DynamicNet(torch.nn.Module):
    """
    定义一个动态神经网络模块，输入层和输出层固定，中间层数量随机（0 - 3 个）且参数共享。
    参数:
        input_dim (int): 输入特征的维度。
        hidden_dim (int): 隐藏层的维度。
        output_dim (int): 输出特征的维度。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(input_dim, hidden_dim)
        self.middle_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 输入和输出层是固定的，中间层个数随机（0~3），参数共享
        h_relu = self.input_linear(x).clamp(min=0)  # 输入层+ReLU
        y_pred = self.output_linear(h_relu)         # 直接输出
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
            y_pred = self.output_linear(h_relu)
        return y_pred

# 批量大小、输入维度、隐藏层维度、输出维度
batch_size = 64
input_dim = 1000
hidden_dim = 83
output_dim = 10

# 随机生成输入和输出数据
torch.manual_seed(0)
x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size, output_dim)

# 创建动态网络模型
model = DynamicNet(input_dim, hidden_dim, output_dim)

# 定义均方误差损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')
# 定义SGD优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for epoch in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()