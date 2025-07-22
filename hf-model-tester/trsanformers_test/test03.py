"""
本脚本主要功能：
----------------
- 加载本地缓存的BERT中文文本分类模型（bert-base-chinese）和分词器。
- 使用 transformers 的 pipeline 工具进行中文文本分类（即判断输入文本属于哪个类别）。
- 自动检测是否有可用GPU，优先使用GPU推理，无GPU则自动切换为CPU。

适用场景：
---------
- 适合在无外网环境下进行大模型本地推理。
- 适合需要自定义模型路径、离线部署、批量文本分类的NLP任务。
- 适合对 HuggingFace Transformers 本地推理流程进行学习和二次开发。

运行说明：
---------
1. 需提前用 test01.py 脚本将模型和分词器下载到本地缓存。
2. 安装依赖：pip install transformers torch
3. 直接运行本脚本即可，支持 GPU/CPU 自动切换。
"""
import os
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

# 指定模型名称，transformers会自动从缓存加载。
# 运行test01.py可以提前下载好。
model_name = "bert-base-chinese"


# 自动检测是否有可用GPU，如果没有则使用CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"当前推理设备: {device}")

# 加载本地BERT分类模型和分词器
# from_pretrained会先检查本地缓存，如果存在就直接加载
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 创建文本分类pipeline，自动选择设备
# device参数: -1为cpu, 0为cuda:0, 1为cuda:1 ...
pipeline_device = 0 if device == "cuda" else -1
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=pipeline_device)

# 对输入文本进行分类
text = "你好，我是一款语言模型"
result = classifier(text)
print(f"输入文本: {text}")
print(f"分类结果: {result}")

# 打印模型结构信息（可选）
# print(model)