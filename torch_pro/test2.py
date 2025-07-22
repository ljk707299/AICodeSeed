import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
# ==============================
# 2. 数据预处理(将数据处理成模型可输入的形状)
# ==============================
print("\n" + "=" * 50)
print("2. 数据预处理示例")
print("=" * 50)


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# 创建模拟数据
num_samples = 100
data = np.random.randn(num_samples, 3, 32, 32)  # 100张32x32的RGB图像
targets = np.random.randint(0, 10, num_samples)  # 0-9的标签

# 定义转换管道
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,)),  # 标准化
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15)  # 随机旋转±15度
])

# 创建数据集和数据加载器
dataset = CustomDataset(data, targets, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# 演示数据加载
print(f"数据集大小: {len(dataset)}")
batch = next(iter(dataloader))
print(f"批数据形状: 输入={batch[0].shape}, 标签={batch[1].shape}")

# 使用TensorDataset的简化方法
tensor_x = torch.randn(100, 5)  # 特征
tensor_y = torch.randint(0, 2, (100,))  # 二分类标签
tensor_dataset = TensorDataset(tensor_x, tensor_y)
dataloader_simple = DataLoader(tensor_dataset, batch_size=10, shuffle=True)
