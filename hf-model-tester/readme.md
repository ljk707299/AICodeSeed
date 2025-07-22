# HuggingFace模型测试工具集（hf-model-tester）

本目录包含 HuggingFace Transformers 相关的本地模型加载/推理测试脚本，以及 HuggingFace API 在线推理测试脚本，适合快速验证模型下载、离线推理、API调用等常见场景。

---

## 目录结构

```
hf-model-tester/
├── trsanformers_test/   # 本地transformers模型加载与推理测试
│   ├── test01.py        # 下载模型和分词器到本地
│   ├── test02.py        # 本地GPT2模型文本生成
│   ├── test03.py        # 本地BERT模型文本分类
│   └── model/           # 存放本地缓存的模型文件
├── API_test/            # HuggingFace API在线推理测试
│   ├── api_test01.py    # 匿名API推理
│   └── api_test02.py    # Token鉴权API推理
└── readme.md            # 本说明文档
```

---

## 1. trsanformers_test/ 本地模型测试

- **test01.py**
  - 功能：将指定HuggingFace模型和分词器下载到本地指定目录，便于离线加载。
  - 用法：直接运行，修改`model_name`和`cache_dir`可下载不同模型。

- **test02.py**
  - 功能：加载本地GPT2模型，使用pipeline进行中文文本生成。
  - 详细说明：
    - 加载本地已下载的GPT2中文语言模型（uer/gpt2-chinese-cluecorpussmall）和分词器。
    - 使用 transformers 的 pipeline 工具进行中文文本生成（即给定一句话自动续写）。
    - 自动检测是否有可用GPU，优先使用GPU推理，无GPU则自动切换为CPU。
    - 适合在无外网环境下进行大模型本地推理，或需要自定义模型路径、离线部署、批量生成文本的NLP任务。
    - 适合对 HuggingFace Transformers 本地推理流程进行学习和二次开发。
    - 运行前需提前用 test01.py 或其它方式将模型和分词器下载到本地，确保 model_dir 路径下包含 config.json、pytorch_model.bin、tokenizer.json 等文件。
    - 安装依赖：pip install transformers torch
    - 直接运行本脚本即可，支持 GPU/CPU 自动切换。

- **test03.py**
  - 功能：加载本地BERT中文分类模型，使用pipeline进行文本分类。
  - 详细说明：
    - 加载本地已下载的BERT中文文本分类模型（bert-base-chinese）和分词器。
    - 使用 transformers 的 pipeline 工具进行中文文本分类（即判断输入文本属于哪个类别）。
    - 自动检测是否有可用GPU，优先使用GPU推理，无GPU则自动切换为CPU。
    - 适合在无外网环境下进行大模型本地推理，或需要自定义模型路径、离线部署、批量文本分类的NLP任务。
    - 适合对 HuggingFace Transformers 本地推理流程进行学习和二次开发。
    - 运行前需提前用 transformers 工具将BERT模型和分词器下载到本地，确保 model_dir 路径下包含 config.json、pytorch_model.bin、tokenizer.json 等文件。
    - 安装依赖：pip install transformers torch
    - 直接运行本脚本即可，支持 GPU/CPU 自动切换。

> **注意事项**：
> - 路径建议使用绝对路径，确保包含config.json等完整模型文件。
> - 运行前请安装`transformers`和`torch`等依赖。
> - 所有脚本均有详细中文注释，便于理解和二次开发。

---

## 2. API_test/ HuggingFace API在线推理

- **api_test01.py**
  - 功能：无需Token，匿名调用HuggingFace推理API，适合公开模型快速测试。
  - 用法：直接运行，修改`API_URL`和输入内容即可。

- **api_test02.py**
  - 功能：使用API Token鉴权，调用HuggingFace推理API，适合私有/高配额场景。
  - 用法：将`API_TOKEN`替换为你自己的Token，修改`API_URL`和输入内容即可。

> **注意事项**：
> - API Token请妥善保管，避免泄露。
> - API有调用频率和配额限制，详见HuggingFace官方文档。

---

## 依赖环境
- Python 3.7+
- transformers >= 4.0
- torch >= 1.7
- requests

安装依赖：
```bash
pip install transformers torch requests
```

---

## 典型应用场景
- 离线部署与推理：本地加载模型，适合无外网环境或大规模推理
- API云推理：无需本地部署，适合快速体验和小规模测试
- 中文NLP模型验证：支持GPT2、BERT等主流中文模型

---

## 参考
- [HuggingFace Transformers文档](https://huggingface.co/docs/transformers)
- [HuggingFace Inference API](https://huggingface.co/inference-api)

---

如有问题欢迎提issue或交流！

- **缓存目录说明**：
  - 默认下载到本地缓存目录：
    - Linux/macOS: ~/.cache/huggingface/hub
    - Windows:    C:\\Users\\你的用户名\\.cache\\huggingface\\hub
  - 你可以通过cache_dir参数自定义下载目录，例如：
    ```python
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall", cache_dir="./my_hf_models")
    ```
