#中文对联

from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained("/Users/lijiakai/code/ai_study/AICodeSeed/GPT2-ZHTuner/model/gpt2-chinese-model/models--uer--gpt2-chinese-couplet/snapshots/91b9465fb1be617f69c6f003b0bd6e6642537bec")
tokenizer = BertTokenizer.from_pretrained("/Users/lijiakai/code/ai_study/AICodeSeed/GPT2-ZHTuner/model/gpt2-chinese-model/models--uer--gpt2-chinese-couplet/snapshots/91b9465fb1be617f69c6f003b0bd6e6642537bec")
print(model)

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model,tokenizer,device="cpu")

#使用text_generator生成文本
#do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时，每次生成的结果都是相同的。
for i in range(3):
    print(text_generator("[CLS]丹 枫 江 冷 人 初 去 -",max_length=21, do_sample=True))