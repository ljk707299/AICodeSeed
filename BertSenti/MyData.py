from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataset(Dataset):
    """
    自定义数据集类，用于加载和处理中文情感分析数据
    """
    def __init__(self,split):
        """
        初始化数据集
        :param split: 数据集分割类型(train/test/validation)
        """
        # 从磁盘加载预处理好的数据集
        self.dataset = load_from_disk("data/ChnSentiCorp")
        # 根据split参数选择对应的数据集分割
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        else:
            print("错误：无效的数据集分割类型！")

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.dataset)

    def __getitem__(self, item):
        """
        获取单个样本
        :param item: 样本索引
        :return: (文本, 标签)元组
        """
        text = self.dataset[item]["text"]  # 获取文本内容
        label = self.dataset[item]["label"]  # 获取情感标签
        return text,label

if __name__ == '__main__':
    dataset = MyDataset("train")
    for data in dataset:
        print(data)