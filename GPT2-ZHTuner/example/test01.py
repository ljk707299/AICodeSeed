#中文白话文文章生成
from transformers import GPT2LMHeadModel,BertTokenizer,TextGenerationPipeline

# 加载模型和分词器
# GPT2 config.json 配置文件的分词工具使用的Bert 
# 使用项目相对路径（假设GPT2模型存放在GPT2-ZHTuner/model目录下）
# 加载GPT2模型（包含完整子目录路径）
# Use absolute path to avoid Hugging Face Hub interpretation
model = GPT2LMHeadModel.from_pretrained("/Users/lijiakai/code/ai_study/AICodeSeed/GPT2-ZHTuner/model/gpt2-chinese-model/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
tokenizer = BertTokenizer.from_pretrained("/Users/lijiakai/code/ai_study/AICodeSeed/GPT2-ZHTuner/model/gpt2-chinese-model/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
print(model)

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model, tokenizer, device="cpu")

#使用text_generator生成文本
#do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时，每次生成的结果都是相同的。
for i in range(3):
    print(text_generator("这是很久之前的事情了,", max_length=100, do_sample=True))