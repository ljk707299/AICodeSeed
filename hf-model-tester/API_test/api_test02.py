import requests
# 使用Token访问HuggingFace在线模型

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
API_TOKEN = "XXX"  # 请替换为你自己的Token
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# 发送POST请求，带上Token和输入文本，获取模型生成结果
response = requests.post(API_URL, headers=headers, json={"inputs": "你好，Hugging face"})
print(response)
