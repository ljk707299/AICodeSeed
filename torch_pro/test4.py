import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
# 使用 'SimHei' 字体来支持中文显示, 'KaiTi', 'FangSong' 等也是可选项
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# ==============================
# 综合应用：线性回归
# ==============================
# 这是一个使用PyTorch进行线性回归的简单示例。
# 它包括：
# 1. 生成带噪声的线性数据
# 2. 定义一个线性回归模型
# 3. 使用均方误差损失和随机梯度下降进行训练
# 4. 可视化原始数据和拟合的直线
print("\n" + "=" * 50)
print("综合应用: 线性回归")
print("=" * 50)

# 优化：自动选择设备 (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备是: {device}")

# --- 1. 生成数据 ---
# 设置随机种子，以便每次运行结果一致，方便调试
torch.manual_seed(42)
# 生成一个从0到10的100个点的等差数列，并将其形状变为 (100, 1)
# .reshape(-1, 1) 表示自动计算行数（此处为100），列数为1
X = torch.linspace(0, 10, 100).reshape(-1, 1)
print(f"输入数据X的形状: {X.shape}")
true_weights = 2.5
true_bias = 1.0

# 根据真实的权重和偏置生成y值，并添加一些随机高斯噪声
# torch.randn(X.size()) 会生成与X形状相同、符合标准正态分布的随机数
# 乘以1.5来控制噪声的大小
y = true_weights * X + true_bias + torch.randn(X.size()) * 1.5

# 将数据和标签移动到指定的设备
X = X.to(device)
y = y.to(device)


# --- 2. 定义模型 ---
# 定义一个简单的线性回归模型
# 它继承自 nn.Module，这是所有神经网络模块的基类
class LinearRegression(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super().__init__()
        # 定义一个线性层，输入特征数为1，输出特征数为1
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # 定义模型的前向传播逻辑
        # 直接将输入x通过线性层
        return self.linear(x)


# --- 3. 训练设置 ---
# 实例化模型并将其移动到指定的设备
model = LinearRegression().to(device)
# 定义损失函数：均方误差（Mean Squared Error）
# 它计算的是预测值和真实值之间差的平方的平均值
criterion = nn.MSELoss()
# 定义优化器：随机梯度下降（Stochastic Gradient Descent）
# model.parameters() 会告诉优化器需要更新哪些参数（权重和偏置）
# lr=0.01 是学习率，控制每次参数更新的步长
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 1000

# --- 4. 训练循环 ---
for epoch in range(epochs):
    # 将模型设置为训练模式
    model.train()
    # 将梯度清零
    # 在每次反向传播之前，都需要清零梯度，否则梯度会累加
    optimizer.zero_grad()
    # 前向传播：将输入数据X传入模型，得到预测值outputs
    outputs = model(X)
    # 计算损失：比较预测值outputs和真实值y
    loss = criterion(outputs, y)
    # 反向传播：根据损失计算梯度
    loss.backward()
    # 更新参数：优化器根据梯度更新模型的权重和偏置
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# --- 5. 结果可视化 ---
# 将模型切换到评估模式
# 这会关闭Dropout和BatchNorm等层，在本例中虽然没有这些层，但是个好习惯
model.eval()

# 使用训练好的模型进行预测时，不需要计算梯度
with torch.no_grad():
    # .cpu() 将数据从GPU移回CPU（如果之前在GPU上）
    # .detach() 将张量从计算图中分离出来，使其不再需要梯度（在no_grad()块中非必需，但是个好习惯）
    # .numpy() 将PyTorch张量转换为NumPy数组，以便matplotlib使用
    predicted = model(X).cpu().numpy()

# 绘制原始数据散点图
# 需要将X和y也移回CPU并转换为NumPy数组
plt.scatter(X.cpu().numpy(), y.cpu().numpy(), label='原始数据')
# 绘制模型拟合的直线
plt.plot(X.cpu().numpy(), predicted, 'r-', label='拟合线')
plt.legend()
# 获取学习到的权重和偏置，并设置图表标题
# .item() 用于从只包含一个值的张量中获取该值
learned_weight = model.linear.weight.item()
learned_bias = model.linear.bias.item()
plt.title(f'最终学到的权重: {learned_weight:.2f}, 偏置: {learned_bias:.2f}')
plt.xlabel("X 值")
plt.ylabel("y 值")
plt.grid(True)
plt.show()
