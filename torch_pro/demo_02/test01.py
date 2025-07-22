import numpy as np

# 批量大小、输入维度、隐藏层维度、输出维度
batch_size = 64
input_dim = 1000
hidden_dim = 100
output_dim = 10

# 随机生成输入和输出数据
x = np.random.randn(batch_size, input_dim)
y = np.random.randn(batch_size, output_dim)

# 随机初始化权重参数
w1 = np.random.randn(input_dim, hidden_dim)
w2 = np.random.randn(hidden_dim, output_dim)

learning_rate = 1e-6
for epoch in range(500):
    # 前向传播：输入x经过第一层线性变换和ReLU激活，再经过第二层线性变换得到预测输出
    h = x.dot(w1)  # 第一层线性变换
    h_relu = np.maximum(h, 0)  # ReLU激活函数
    y_pred = h_relu.dot(w2)  # 第二层线性变换，得到预测值

    # 计算损失函数（均方误差）
    loss = np.square(y_pred - y).sum()
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # 反向传播：计算损失对各参数的梯度
    grad_y_pred = 2.0 * (y_pred - y)  # 损失对y_pred的梯度
    grad_w2 = h_relu.T.dot(grad_y_pred)  # 损失对w2的梯度
    grad_h_relu = grad_y_pred.dot(w2.T)  # 损失对h_relu的梯度
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0  # ReLU对负值的梯度为0
    grad_w1 = x.T.dot(grad_h)  # 损失对w1的梯度

    # 更新参数（梯度下降）
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2