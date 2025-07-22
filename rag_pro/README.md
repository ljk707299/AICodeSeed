# RAG Pro 系统

## 系统概述

RAG Pro 是一个高级检索增强生成系统，使用 MultiQueryRetriever 技术来提高检索准确性。相比基础RAG系统，Pro版本具有更强的检索能力和更好的用户体验。

## 主要特性

### 🚀 **多查询检索**
- **自动查询扩展**：系统会自动生成多个相关查询
- **提高召回率**：通过多角度检索，找到更多相关文档
- **智能去重**：自动合并和去重搜索结果

### 🔍 **高级向量搜索**
- **语义相似度**：基于深度学习的语义理解
- **多维度匹配**：考虑语义、关键词等多个维度
- **智能排序**：根据相关性自动排序结果

### 🤖 **智能模型集成**
- **通义千问**：使用阿里云最新的通义千问模型
- **稳定可靠**：经过优化的模型配置
- **中文优化**：专门针对中文内容优化

### 📊 **完善的日志系统**
- **详细记录**：记录所有操作和错误信息
- **便于调试**：提供完整的调试信息
- **性能监控**：监控系统性能指标

### 🧠 **Assistant智能体**
- **文件理解**：直接读取PDF、Word等文档
- **智能问答**：基于文档内容进行准确回答
- **工具集成**：支持多种工具和函数调用
- **日志分析**：详细的检索过程日志记录

## 技术架构

```
用户查询 → MultiQueryRetriever → 查询扩展 → 向量搜索 → 结果合并 → 智能排序 → 最终结果
    ↓              ↓                ↓           ↓          ↓          ↓
  输入处理      多查询生成        相似度计算    FAISS索引   去重合并   相关性排序
```

### Assistant智能体架构
```
用户问题 → Assistant智能体 → 文档解析 → 内容检索 → 智能回答 → 日志记录
    ↓            ↓              ↓          ↓          ↓          ↓
  问题输入    文件读取        文本提取   语义匹配   答案生成   过程追踪
```

## 系统组件

### 1. MultiQueryRetriever (main.py)
- **功能**：基于LangChain的多查询检索系统
- **特点**：自动生成多个相关查询，提高检索覆盖率
- **适用场景**：复杂问题检索、高精度搜索

### 2. Assistant智能体 (rag_qwen.py)
- **功能**：基于qwen_agent的智能文档问答系统
- **特点**：直接读取文档，无需预处理向量化
- **适用场景**：快速文档问答、实时文件处理

## 安装和配置

### 1. 环境要求
- Python 3.8+
- 阿里百炼API密钥

### 2. 安装依赖

#### MultiQueryRetriever依赖
```bash
pip install langchain langchain-community faiss-cpu dashscope
```

#### Assistant智能体依赖
```bash
pip install qwen-agent dashscope
```

### 3. 配置API密钥
```bash
export DASHSCOPE_API_KEY="your_api_key"
# 或
export ALI_API_KEY="your_api_key"
```

### 4. 准备知识库
确保 `./knowledge_base` 目录存在并包含向量数据库文件（用于MultiQueryRetriever）。

## 使用方法

### MultiQueryRetriever使用 (main.py)

#### 基本使用
```python
from main import main

# 运行系统
main()
```

#### 自定义查询
```python
from main import initialize_models, load_vectorstore, create_retriever, search_documents

# 初始化系统
api_key = "your_api_key"
llm, embeddings = initialize_models(api_key)
vectorstore = load_vectorstore(embeddings)
retriever = create_retriever(vectorstore, llm)

# 执行查询
query = "你的问题"
results = search_documents(retriever, query)
```

### Assistant智能体使用 (rag_qwen.py)

#### 基本使用
```python
# 直接运行脚本
python rag_qwen.py
```

#### 自定义配置
```python
from rag_qwen import LogCapture, Assistant

# 配置LLM
llm_cfg = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    'api_key': os.getenv("ALI_API_KEY"),
    'generate_cfg': {'top_p': 0.8}
}

# 创建智能体
bot = Assistant(
    llm=llm_cfg,
    system_message="你是一个专业的文档问答助手",
    function_list=[],
    files=['./your_document.pdf']
)

# 进行问答
messages = [{'role': 'user', 'content': '你的问题'}]
for response in bot.run(messages=messages):
    print(response[0]['content'])
```

## 核心组件

### 1. MultiQueryRetriever
- **功能**：自动生成多个相关查询
- **优势**：提高检索覆盖率和准确性
- **配置**：支持自定义查询生成策略

### 2. FAISS向量数据库
- **功能**：高效的向量相似度搜索
- **优势**：快速、准确、可扩展
- **特性**：支持GPU加速（可选）

### 3. 通义千问模型
- **功能**：自然语言理解和生成
- **优势**：中文支持优秀，性能稳定
- **配置**：使用turbo版本，平衡性能和速度

### 4. Assistant智能体
- **功能**：智能文档问答和工具调用
- **优势**：
  - 直接读取多种格式文档（PDF、Word、TXT等）
  - 无需预先向量化处理
  - 支持实时文件处理
  - 详细的检索过程日志
- **配置**：支持自定义系统提示词和工具列表

### 5. LogCapture日志捕获器
- **功能**：捕获智能体运行过程中的详细日志
- **优势**：
  - 实时监控检索过程
  - 分析文档处理流程
  - 调试和优化系统性能
- **输出**：分类显示关键词提取、文档处理、检索相关等信息

## 性能优化

### 1. 查询优化
- **查询扩展**：自动生成相关查询
- **智能过滤**：过滤无关结果
- **结果排序**：按相关性排序

### 2. 向量搜索优化
- **索引优化**：使用高效的FAISS索引
- **批量处理**：支持批量查询
- **缓存机制**：缓存常用查询结果

### 3. 模型优化
- **模型选择**：使用适合的模型版本
- **参数调优**：优化模型参数
- **错误处理**：完善的错误恢复机制

### 4. Assistant智能体优化
- **文档预处理**：智能文档解析和分块
- **内存管理**：高效的内存使用
- **并发处理**：支持多文档并行处理

## 错误处理

### 常见问题

#### 1. API密钥错误
```
错误: 未配置API密钥
解决: 设置DASHSCOPE_API_KEY或ALI_API_KEY环境变量
```

#### 2. 知识库路径错误
```
错误: 知识库路径不存在
解决: 确保./knowledge_base目录存在并包含正确的文件
```

#### 3. 向量维度不匹配
```
错误: AssertionError: assert d == self.d
解决: 确保使用相同的嵌入模型版本
```

#### 4. LangChain弃用警告
```
警告: LangChainDeprecationWarning
解决: 已使用最新的invoke方法替代get_relevant_documents
```

#### 5. qwen_agent导入错误
```
错误: ModuleNotFoundError: No module named 'qwen_agent'
解决: 安装qwen-agent包：pip install qwen-agent
```

#### 6. 文档读取失败
```
错误: 无法读取指定文档
解决: 检查文档路径和格式是否正确
```

### 调试建议
1. 查看日志文件 `rag_pro.log` 或 `rag_qwen.log`
2. 检查API密钥配置
3. 验证知识库完整性
4. 确认模型版本兼容性
5. 检查文档文件是否存在且可读

## 系统对比

| 特性 | MultiQueryRetriever | Assistant智能体 |
|------|-------------------|-----------------|
| 文档处理 | 需要预先向量化 | 直接读取文档 |
| 检索方式 | 多查询检索 | 智能文档检索 |
| 处理速度 | 快速（已向量化） | 中等（实时处理） |
| 内存占用 | 较低 | 中等 |
| 适用场景 | 大规模文档检索 | 实时文档问答 |
| 日志详细度 | 基础日志 | 详细检索日志 |
| 部署复杂度 | 简单 | 中等 |

## 使用场景

### 1. 复杂问题检索
- 多角度问题分析
- 跨领域信息检索
- 深度内容挖掘

### 2. 高精度搜索
- 专业文档检索
- 技术资料搜索
- 学术研究支持

### 3. 企业知识管理
- 内部文档检索
- 政策法规查询
- 技术文档搜索

### 4. 实时文档问答
- 新文档快速问答
- 临时文件处理
- 动态内容检索

### 5. 文档分析调试
- 检索过程分析
- 文档处理优化
- 系统性能调优

## 扩展功能

### 1. 支持更多检索器
- BM25检索器
- 混合检索器
- 自定义检索器

### 2. 支持更多向量数据库
- ChromaDB
- Pinecone
- Weaviate

### 3. 支持更多LLM
- OpenAI GPT
- Claude
- 本地模型

### 4. 支持更多文档格式
- PDF、Word、TXT
- Markdown、HTML
- 图片OCR识别

### 5. 支持更多工具
- 计算器
- 网络搜索
- 数据库查询

## 许可证

MIT License


---

**开始使用RAG Pro，体验更强大的检索能力！** 🚀
