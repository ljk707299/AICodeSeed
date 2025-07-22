"""
本脚本为模型预热/下载脚本。
----------------
- 负责将后续测试（test02, test03）所需的模型和分词器下载到HuggingFace的本地缓存中。
- 运行一次本脚本后，其他脚本即可实现离线加载。

下载内容：
---------
1. uer/gpt2-chinese-cluecorpussmall (用于 test02.py)
2. bert-base-chinese (用于 test03.py)

缓存目录说明：
-------------
- 默认下载到本地缓存目录：
    - Linux/macOS: ~/.cache/huggingface/hub
    - Windows:    C:\\Users\\你的用户名\\.cache\\huggingface\\hub
- 你可以通过cache_dir参数自定义下载目录，见下方示例。

运行说明：
---------
- 直接运行即可，脚本会自动下载并缓存模型。
- 请确保网络连接正常。
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertForSequenceClassification
import os

# 打印默认缓存目录
print("HuggingFace模型默认缓存目录:")
print(os.path.expanduser("~/.cache/huggingface/hub"))

# --- 1. 下载 GPT-2 中文模型 (用于 test02.py) ---
print("="*50)
print("开始下载 gpt2-chinese-cluecorpussmall 模型...")
gpt_model_name = "uer/gpt2-chinese-cluecorpussmall"
# 下载模型和分词器，它们会自动保存到HuggingFace的默认缓存目录
AutoModelForCausalLM.from_pretrained(gpt_model_name)
AutoTokenizer.from_pretrained(gpt_model_name)
print(f"'{gpt_model_name}' 下载完成。")
print("="*50)

# --- 2. 下载 BERT 中文模型 (用于 test03.py) ---
print("\n"+"="*50)
print("开始下载 bert-base-chinese 模型...")
bert_model_name = "bert-base-chinese"
# 下载模型和分词器
BertForSequenceClassification.from_pretrained(bert_model_name)
BertTokenizer.from_pretrained(bert_model_name)
print(f"'{bert_model_name}' 下载完成。")
print("="*50)

# --- 3. 示例：如何指定自定义下载目录 ---
custom_dir = os.path.abspath("./my_hf_models")
print(f"\n示例：将模型下载到自定义目录: {custom_dir}")
# 只做演示，实际可根据需要取消注释
# AutoModelForCausalLM.from_pretrained(gpt_model_name, cache_dir=custom_dir)
# AutoTokenizer.from_pretrained(gpt_model_name, cache_dir=custom_dir)

print("\n所有模型下载完成！")