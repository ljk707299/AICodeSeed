"""
数据测试脚本，用于验证数据集加载功能
"""
from datasets import load_dataset, load_from_disk

# 测试数据集加载功能
def test_data_loading():
    """
    测试从磁盘加载数据集的功能
    """
    # 从磁盘加载预处理好的数据集
    datasets = load_from_disk("data/ChnSentiCorp")
    print("数据集加载成功，结构如下:")
    print(datasets)

    # 测试集数据预览
    print("\n测试集数据示例:")
    test_data = datasets["test"]
    for i, data in enumerate(test_data):
        if i >= 5:  # 只打印前5条
            break
        print(f"样本{i+1}: {data}")

    # 新增CSV数据加载测试
    print("\nCSV数据加载测试:")
    csv_dataset = load_dataset(path="csv", data_files="data/hermes-function-calling-v1.csv")
    print("CSV数据集加载成功，结构如下:")
    print(csv_dataset)
    print("\nCSV数据示例:")
    for i, data in enumerate(csv_dataset["train"]):
        if i >= 3:  # 只打印前3条
            break
        print(f"样本{i+1}: {data}")

    # 在线加载数据测试(注释状态)
    """
    # 从Hugging Face加载数据集
    dataset = load_dataset(path="NousResearch/hermes-function-calling-v1", cache_dir="data/")
    print(dataset)
    
    # 转为csv格式
    dataset.to_csv(path_or_buf="data/ChnSentiCorp.csv")
    """

if __name__ == '__main__':
    test_data_loading()
