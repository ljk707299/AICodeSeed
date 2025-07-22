import requests

# HuggingFace推理API地址（无需Token，匿名访问）
API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"

# 发送POST请求，输入prompt文本，获取模型生成结果
response = requests.post(API_URL, json={"input": "你好,hugging face"})
print(response.json())