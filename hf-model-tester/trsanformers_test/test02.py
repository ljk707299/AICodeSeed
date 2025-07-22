"""
本脚本主要功能：
----------------
- 加载本地缓存的GPT2中文语言模型（uer/gpt2-chinese-cluecorpussmall）和分词器。
- 使用 transformers 的 pipeline 工具进行中文文本生成（即给定一句话自动续写）。
- 自动检测是否有可用GPU，优先使用GPU推理，无GPU则自动切换为CPU。

适用场景：
---------
- 适合在无外网环境下进行大模型本地推理。
- 适合需要自定义模型路径、离线部署、批量生成文本的NLP任务。
- 适合对 HuggingFace Transformers 本地推理流程进行学习和二次开发。

运行说明：
---------
1. 需提前用 test01.py 脚本将模型和分词器下载到本地缓存。
2. 安装依赖：pip install transformers torch
3. 直接运行本脚本即可，支持 GPU/CPU 自动切换。
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 指定模型名称，transformers会自动从缓存加载。
# 运行test01.py可以提前下载好。
model_name = "uer/gpt2-chinese-cluecorpussmall"

# 自动检测是否有可用GPU，如果没有则使用CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"当前推理设备: {device}")

# 加载本地GPT2模型和分词器
# from_pretrained会先检查本地缓存，如果存在就直接加载
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建文本生成pipeline，自动选择设备
# device参数: -1为cpu, 0为cuda:0, 1为cuda:1 ...
pipeline_device = 0 if device == "cuda" else -1
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=pipeline_device)

# 使用pipeline进行文本生成
output = generator(
    "你好，我是一款语言模型，",  # 输入的prompt文本
    max_length=50,                # 生成文本的最大长度（token数）
    num_return_sequences=1,       # 返回的生成序列数
    truncation=True,              # 超长输入是否截断
    temperature=0.7,              # 控制生成的多样性，越低越保守，越高越随机
    top_k=50,                     # 每步只考虑概率最高的前k个词
    top_p=0.9,                    # nucleus采样，累计概率达到p的词汇集合中采样
    clean_up_tokenization_spaces=True # 是否清理多余空格
)
print(output)