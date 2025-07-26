
## 模型说明
本项目使用的预训练模型基于[uer/gpt2-chinese-*](https://huggingface.co/uer)系列，包括以下变体：
- cluecorpussmall: 通用中文模型
- lyric: 中文歌词生成模型
- ancient: 古文生成模型
- couplet: 对联生成模型
- poem: 诗歌生成模型

模型文件存储在`model/gpt2-chinese-model/`目录下，采用Hugging Face的标准模型格式，包含以下关键文件：
- pytorch_model.bin/model.safetensors: 模型权重文件
- config.json: 模型配置文件
- vocab.json/merges.txt: 分词器文件
- tokenizer_config.json: 分词器配置

## 使用方法
### 1. 数据准备
将自定义数据集放入`data/`目录，格式为纯文本文件，每行一段文本。项目已提供中文诗歌数据集`chinese_poems.txt`作为示例。

### 2. 模型训练
```bash
python train.py
```
训练参数可在`train.py`中调整，主要包括：
- 学习率(learning_rate)
- 训练轮次(num_train_epochs)
- 批处理大小(batch_size)
- 最大序列长度(max_seq_length)
训练结果将保存在`params/`目录下

### 3. 文本生成
使用示例脚本进行文本生成：
```bash
# 基础文本生成
python example/test01.py

# 诗歌生成
python example/test05.py
```
或使用检测模块进行交互式生成：
```bash
python detect.py
python detect02.py
```

## 路径配置说明
模型加载路径需使用绝对路径，macOS系统默认路径格式为：
```python
/Users/lijiakai/code/ai_study/AICodeSeed/GPT2-ZHTuner/model/gpt2-chinese-model/models--uer--[model-name]/snapshots/[hash]/
```
不同系统路径格式差异：
- Windows: `D:\model\gpt2-chinese-model\...`
- macOS/Linux: `/path/to/model/gpt2-chinese-model/...`

## 常见问题
### Q: 模型加载失败怎么办？
A: 确保模型路径正确且完整，检查是否包含所有必要的模型文件。路径格式需符合当前操作系统要求，建议使用绝对路径。

### Q: 训练时报错"CUDA out of memory"？
A: 尝试减小批处理大小(batch_size)或最大序列长度(max_seq_length)，或使用更小的模型变体。

## 致谢
- 基于[Hugging Face Transformers](https://huggingface.co/transformers)库开发
- 预训练模型来自[uer](https://huggingface.co/uer)的GPT2-Chinese项目



---
