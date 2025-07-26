from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("/Users/lijiakai/code/ai_study/AICodeSeed/GPT2-ZHTuner/model/gpt2-chinese-model/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained("/Users/lijiakai/code/ai_study/AICodeSeed/GPT2-ZHTuner/model/gpt2-chinese-model/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")

# 加载我们自己训练的权重（中文古诗词）
# Load model with CPU mapping
device = torch.device('cpu')
# 加载状态字典并应用到模型
state_dict = torch.load('params/net.pt', map_location=device)
model.load_state_dict(state_dict)

# 使用系统自带的pipeline工具生成内容
pipeline = TextGenerationPipeline(model, tokenizer, device=-1)

for i in range(5):
    print(pipeline("白日", max_length=24,do_sample=True))