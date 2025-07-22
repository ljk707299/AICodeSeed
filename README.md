# AI代码种子项目 (AICodeSeed)

## 项目概述

AICodeSeed 是一个面向AI开发者和学习者的综合性项目，涵盖了向量数据库、RAG（检索增强生成）、深度学习基础与PyTorch实战等多个模块。项目旨在帮助用户系统性掌握现代AI技术的核心原理与工程实践。

---

## 项目结构

```
AICodeSeed/
├── README.md              # 项目总览文档（本文件）
├── rag/                   # RAG系统模块（检索增强生成）
├── Embeddings/            # 向量数据库与嵌入示例
├── torch_pro/             # PyTorch深度学习实战与基础
│   └── demo_02/           # PyTorch基础与进阶代码示例
└── venv/                  # Python虚拟环境
```

---

## 各模块功能简介

### 1. rag/ —— 检索增强生成（RAG）系统
- **功能**：基于PDF文档的智能问答系统，集成向量数据库与大语言模型。
- **主要特性**：
  - PDF文档解析与分块
  - 向量化存储（FAISS等）与相似度检索
  - 智能问答（支持通义千问、OpenAI GPT等）
  - 知识库持久化与高效加载
- **典型用法**：
  ```python
  from rag.main import RAGSystem, RAGConfig
  config = RAGConfig(chunk_size=512, llm_model="qwen-turbo")
  rag_system = RAGSystem(config)
  knowledge_base = rag_system.process_pdf("your_document.pdf")
  answer = rag_system.ask_question("你的问题")
  ```
- **适合人群**：希望构建企业级智能问答、文档检索系统的开发者

### 2. Embeddings/ —— 向量数据库与嵌入示例
- **功能**：多种主流向量数据库的完整用法与对比，包括：
  - 阿里百炼（国内平台，免费额度大）
  - ChromaDB（本地开源）
  - FAISS（高性能本地库）
  - Pinecone（云原生服务）
- **内容**：
  - 各平台API调用示例
  - 安装与配置指南
  - 代码注释与性能对比
- **适合人群**：向量检索、嵌入存储、AI搜索等场景开发者

### 3. torch_pro/demo_02/ —— PyTorch基础与进阶实战
- **功能**：系统梳理神经网络与PyTorch开发流程，涵盖：
  - Numpy手写神经网络
  - PyTorch张量与自动求导
  - nn.Module与nn.Sequential建模
  - 优化器（SGD/Adam）与动态网络
- **典型用法**：
  - 逐步运行 test01.py ~ test09.py，理解从底层到高阶的神经网络实现
- **PyTorch训练流程简述**：
  1. 数据准备（张量/Dataset/DataLoader）
  2. 模型定义（nn.Module/nn.Sequential）
  3. 损失函数与优化器
  4. 训练循环（前向传播→损失→梯度清零→反向传播→参数更新）
  5. 评估与推理（model.eval(), torch.no_grad()）
- **适合人群**：深度学习/PyTorch初学者、需要理解底层原理的开发者

---

## 快速开始

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd AICodeSeed
   ```
2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows
   ```
3. **选择模块并安装依赖**
   - RAG系统：`cd rag && pip install -r requirements.txt`
   - 向量数据库：`cd Embeddings && pip install -r requirements.txt`
   - PyTorch实战：`cd torch_pro/demo_02`（如需GPU请提前安装CUDA版PyTorch）
4. **配置API密钥**（如需调用阿里百炼、OpenAI等）
   ```bash
   export DASHSCOPE_API_KEY="your_api_key"
   export OPENAI_API_KEY="your_api_key"
   ```
5. **运行示例脚本**
   - 见各子目录README.md

---

## 学习路径建议

### 初学者
1. 从 `torch_pro/demo_02/` 开始，理解神经网络与PyTorch基础
2. 运行 `Embeddings/` 示例，掌握向量数据库基本用法
3. 阅读注释，尝试修改参数、结构

### 进阶用户
1. 深入 `rag/`，学习RAG系统设计与工程实现
2. 理解各组件职责与交互，尝试扩展功能

### 开发者
1. 研究整体架构，学习最佳工程实践
2. 优化性能，扩展新平台支持

---

## 技术亮点
- 现代化架构与模块化设计
- 多平台、多模型支持
- 完整的中文文档与详细注释
- 丰富的实战案例与对比分析
- 适合教学、科研、工程落地

---

## 贡献指南

欢迎贡献代码和文档！
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

---

## 许可证

MIT License

---

## 致谢

感谢以下开源项目和平台的支持：
- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [阿里百炼](https://bailian.console.aliyun.com/)
- [OpenAI](https://openai.com/)

---

**开启你的AI学习与工程实践之旅！** 🚀
