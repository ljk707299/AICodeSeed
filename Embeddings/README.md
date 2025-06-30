# 向量数据库与文本嵌入示例项目

本项目演示了三种主流向量数据库（ChromaDB、FAISS、Pinecone）以及阿里百炼API的文本嵌入与相似度计算的完整流程，适合AI/NLP开发者快速上手和对比选型。

---

## 📁 目录结构

```
Embeddings/
├── README.md              # 本说明文档
├── INSTALL.md             # 安装与环境配置指南
├── main.py                # 阿里百炼API嵌入向量示例
├── chroma_main.py         # ChromaDB向量数据库示例
├── faiss_example.py       # FAISS本地向量库示例
└── pinecone_example.py    # Pinecone云端向量数据库示例
```

---

## 📝 各脚本功能简介

- **main.py**：调用阿里百炼API生成文本嵌入，支持余弦/欧氏距离相似度计算，适合中文场景。
- **chroma_main.py**：本地ChromaDB数据库，支持文档、元数据存储与相似度检索，适合原型开发和小规模应用。
- **faiss_example.py**：FAISS高性能本地向量库，支持多种索引类型，适合大规模、对性能有要求的场景。
- **pinecone_example.py**：Pinecone云端向量数据库，支持托管、自动扩展，适合生产环境和大数据量。

---

## 🚀 环境准备与依赖安装

1. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows用 venv\Scripts\activate
   ```
2. **安装依赖**
   ```bash
   pip install numpy openai chromadb faiss-cpu pinecone-client
   ```
3. **配置API密钥（如需云服务）**
   ```bash
   export ALI_API_KEY=你的阿里百炼key
   export ALI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   export OPENAI_API_KEY=你的OpenAI key
   export PINECONE_API_KEY=你的Pinecone key
   ```

---

## 📚 各向量数据库/服务使用流程

### 1. 阿里百炼API（main.py）
- 初始化OpenAI兼容客户端，配置API Key和Base URL
- 使用`get_embeddings`函数获取文本嵌入
- 计算余弦/欧氏距离，输出相似度结果
- 适合高质量中文嵌入、跨语言场景

### 2. ChromaDB（chroma_main.py）
- 创建本地ChromaDB客户端和集合
- 添加文档、元数据，自动生成嵌入
- 支持基于文本、元数据、内容的多种检索
- 适合原型开发、小型项目、元数据丰富场景

### 3. FAISS（faiss_example.py）
- 生成模拟向量或加载实际嵌入
- 支持Flat、IVF、HNSW等多种索引类型
- 添加向量后可高效检索相似向量
- 适合大规模、高性能本地检索

### 4. Pinecone（pinecone_example.py）
- 云端托管，需API Key和付费计划
- 支持索引创建、向量插入、相似度检索、元数据过滤
- 示例脚本支持本地模拟向量演示
- 适合生产环境、自动扩展、企业级需求

---

## 🛠️ 运行方法

```bash
python main.py            # 阿里百炼API示例
python chroma_main.py     # ChromaDB本地数据库示例
python faiss_example.py   # FAISS本地库示例
python pinecone_example.py # Pinecone云端数据库/本地模拟示例
```

---

## 🔄 选型建议

| 场景         | 推荐方案   | 理由                     |
|--------------|------------|--------------------------|
| 原型/小项目  | ChromaDB   | 易用、支持元数据         |
| 研究/实验    | FAISS      | 免费、高性能              |
| 生产/大数据  | Pinecone   | 托管、可扩展、企业级      |
| 中文文本      | 阿里百炼   | 优秀中文理解、云服务      |

---

## 📊 性能与特性对比

| 指标     | 阿里百炼 | ChromaDB | FAISS | Pinecone |
|----------|---------|----------|-------|----------|
| 搜索速度 | 中等    | 慢       | 快    | 快       |
| 易用性   | 高      | 高       | 低    | 中等     |
| 成本     | 按量计费| 免费     | 免费  | 按量计费 |
| 部署     | 云端    | 本地     | 本地  | 云端     |
| 元数据   | 支持    | 支持     | 不支持| 支持     |

---

## ❓ 常见问题

- **FAISS安装失败**：请用`pip install faiss-cpu`，Apple Silicon同样适用。
- **Pinecone报错**：请确保API Key有效，区域选择正确，免费账户有区域和配额限制。
- **ChromaDB警告**：`Failed to send telemetry event`为遥测警告，可忽略。
- **API配额/速率限制**：请检查API账户配额，或使用本地模拟向量。

---

## 📖 进一步学习

- [阿里百炼API文档](https://help.aliyun.com/zh/dashscope/)
- [ChromaDB官方文档](https://docs.trychroma.com/)
- [FAISS官方文档](https://github.com/facebookresearch/faiss)
- [Pinecone官方文档](https://docs.pinecone.io/)
- [向量数据库对比](https://zilliz.com/comparison)

---

## 🤝 贡献与反馈

欢迎提交Issue、PR或建议，共同完善本项目！

---

## 📄 许可证

本项目采用MIT许可证。 
