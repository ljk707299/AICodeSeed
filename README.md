# AI代码种子项目 (AICodeSeed)

## 项目概述

这是一个综合性的AI学习和实践项目，包含向量数据库应用和RAG（检索增强生成）系统的完整实现。项目旨在帮助开发者学习和实践现代AI技术，特别是向量数据库和大语言模型的应用。

## 项目结构

```
AICodeSeed/
├── README.md              # 项目总览文档
├── rag/                   # RAG系统模块
│   ├── main.py           # RAG系统主程序
│   ├── README.md         # RAG系统详细文档
│   ├── requirements.txt  # RAG系统依赖
│   ├── setup.py          # 快速设置脚本
│   ├── example.py        # 使用示例
│   └── test_tongyi.py    # 通义千问测试
├── Embeddings/           # 向量数据库示例模块
│   ├── README.md         # 向量数据库文档
│   ├── INSTALL.md        # 安装指南
│   ├── main.py           # 阿里百炼示例
│   ├── chroma_main.py    # ChromaDB示例
│   ├── faiss_example.py  # FAISS示例
│   └── pinecone_example.py # Pinecone示例
└── venv/                 # Python虚拟环境
```

## 主要功能模块

### 1. RAG系统 (rag/)

基于PDF文档的智能问答系统，支持：

- **PDF文档处理**: 自动提取文本和页码信息
- **文本分块**: 智能分割长文档为适合向量化的小块
- **向量化存储**: 使用FAISS进行高效的向量存储
- **相似度搜索**: 基于语义的文档检索
- **智能问答**: 集成大语言模型进行自然语言问答
- **知识库持久化**: 支持保存和加载知识库

**支持的模型**:
- 嵌入模型: 阿里百炼、OpenAI
- 大语言模型: 通义千问、OpenAI GPT

### 2. 向量数据库示例 (Embeddings/)

提供多种向量数据库的完整实现示例：

- **阿里百炼**: 国内领先的AI平台，免费额度大
- **ChromaDB**: 开源的向量数据库，适合本地部署
- **FAISS**: Facebook开发的向量搜索库，性能优异
- **Pinecone**: 云原生向量数据库，易于使用

## 快速开始

### 环境要求

- Python 3.8+
- 至少一个AI平台的API密钥

### 1. 克隆项目

```bash
git clone <repository-url>
cd AICodeSeed
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 3. 选择要使用的模块

#### 使用RAG系统

```bash
cd rag
pip install -r requirements.txt
python setup.py  # 快速设置
python main.py   # 运行测试
```

#### 使用向量数据库示例

```bash
cd Embeddings
pip install -r requirements.txt
python main.py  # 运行阿里百炼示例
```

### 4. 配置API密钥

#### 阿里百炼API（推荐）

1. 访问 [阿里百炼官网](https://bailian.console.aliyun.com/)
2. 注册并登录账户
3. 创建应用获取API密钥
4. 设置环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key"
# 或
export ALI_API_KEY="your_api_key"
```

#### OpenAI API

```bash
export OPENAI_API_KEY="your_api_key"
```

## 使用指南

### RAG系统使用

```python
from rag.main import RAGSystem, RAGConfig

# 创建配置
config = RAGConfig(
    chunk_size=512,
    llm_model="qwen-turbo"
)

# 创建RAG系统
rag_system = RAGSystem(config)

# 处理PDF文档
knowledge_base = rag_system.process_pdf("your_document.pdf")

# 智能问答
answer = rag_system.ask_question("你的问题")
```

### 向量数据库使用

每个向量数据库示例都包含：
- 完整的安装指南
- 详细的代码注释
- 性能对比分析
- 故障排除指南

## 技术特性

### 1. 现代化架构
- 使用最新的LangChain API
- 无弃用警告
- 模块化设计
- 完整的错误处理

### 2. 多平台支持
- 支持多种AI平台
- 跨平台兼容
- 灵活的配置选项

### 3. 性能优化
- 智能文本分块
- 高效的向量搜索
- 懒加载模式
- 成本跟踪

### 4. 用户友好
- 详细的中文文档
- 完整的示例代码
- 友好的错误信息
- 快速设置脚本

## 学习路径

### 初学者
1. 从 `Embeddings/` 模块开始，了解向量数据库基础
2. 运行各个示例，理解不同平台的特点
3. 阅读代码注释，学习实现原理

### 进阶用户
1. 深入 `rag/` 模块，学习RAG系统设计
2. 理解各个组件的职责和交互
3. 尝试修改配置，优化性能

### 开发者
1. 研究代码架构，学习最佳实践
2. 扩展功能，添加新的向量数据库支持
3. 优化性能，改进用户体验

## 常见问题

### Q: 如何选择适合的向量数据库？
A: 
- **本地部署**: ChromaDB、FAISS
- **云服务**: Pinecone、阿里百炼
- **免费使用**: 阿里百炼（免费额度大）
- **高性能**: FAISS

### Q: API配额超限怎么办？
A: 
- 检查账户余额
- 等待配额重置
- 使用其他API服务
- 优化请求频率

### Q: 如何处理大文档？
A: 
- 调整文本块大小
- 使用合适的重叠设置
- 分批处理
- 使用本地向量数据库

## 贡献指南

欢迎贡献代码和文档！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License

## 更新日志

### v2.1 (2024-06-29)
- ✅ 完全修复LangChain弃用警告
- ✅ 集成通义千问替代DeepSeek
- ✅ 解决API兼容性问题
- ✅ 抑制FAISS GPU警告
- ✅ 改进LLM响应处理

### v2.0 (2024-06-29)
- ✅ 重构RAG系统架构
- ✅ 添加完整的错误处理
- ✅ 支持多种API服务
- ✅ 添加详细中文注释

### v1.0 (2024-06-29)
- ✅ 基础RAG系统实现
- ✅ 多种向量数据库示例
- ✅ 完整的文档和示例

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

## 致谢

感谢以下开源项目和平台的支持：

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [阿里百炼](https://bailian.console.aliyun.com/)
- [OpenAI](https://openai.com/)

---

**开始你的AI学习之旅吧！** 🚀
