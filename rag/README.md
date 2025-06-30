# RAG系统使用指南

## 系统概述

这是一个基于PDF文档的RAG (Retrieval-Augmented Generation) 系统，支持：
- PDF文档文本提取和页码记录
- 文本分块处理
- 向量化存储（支持FAISS）
- 相似度搜索
- 知识库持久化
- 智能问答（支持多种LLM）

## 代码流程说明

### 整体架构流程

```
PDF文档 → 文本提取 → 文本分块 → 向量化 → 存储 → 搜索 → 问答生成
   ↓         ↓         ↓         ↓       ↓      ↓       ↓
页码记录   文本清理   重叠处理   嵌入API   FAISS  相似度    LLM API
```

### 详细流程说明

#### 1. 文档处理阶段 (Document Processing)

```python
# 流程：PDF文档 → 文本提取 → 页码记录
def process_pdf(self, pdf_path: str) -> KnowledgeBase:
    """
    处理PDF文档的完整流程：
    1. 加载PDF文件
    2. 逐页提取文本
    3. 记录页码信息
    4. 文本预处理
    5. 返回知识库对象
    """
```

**具体步骤：**
- **PDF加载**: 使用PyPDF2加载PDF文件
- **页面遍历**: 逐页提取文本内容
- **页码记录**: 为每个文本块记录原始页码
- **文本清理**: 去除多余空白和特殊字符
- **编码处理**: 确保文本编码正确

#### 2. 文本分块阶段 (Text Chunking)

```python
# 流程：原始文本 → 智能分块 → 重叠处理
def _split_text(self, text: str) -> List[str]:
    """
    文本分块的核心逻辑：
    1. 按分隔符分割文本
    2. 控制块大小
    3. 添加重叠内容
    4. 保持语义完整性
    """
```

**分块策略：**
- **分隔符优先**: 优先在自然分隔符处分割
- **大小控制**: 确保每个块不超过指定大小
- **重叠处理**: 相邻块之间保持一定重叠
- **语义保持**: 尽量保持语义单元的完整性

#### 3. 向量化阶段 (Vectorization)

```python
# 流程：文本块 → 嵌入API调用 → 向量生成
def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
    """
    向量化处理流程：
    1. 选择嵌入模型
    2. 批量调用API
    3. 错误处理和重试
    4. 返回向量列表
    """
```

**API选择逻辑：**
1. **优先使用阿里百炼**: 检查 `DASHSCOPE_API_KEY`
2. **降级到OpenAI**: 如果阿里百炼不可用
3. **错误处理**: API调用失败时的重试机制
4. **批量处理**: 减少API调用次数

#### 4. 存储阶段 (Storage)

```python
# 流程：向量数据 → FAISS索引 → 持久化存储
def _create_faiss_index(self, embeddings: List[List[float]]) -> Any:
    """
    向量存储流程：
    1. 创建FAISS索引
    2. 添加向量数据
    3. 构建搜索索引
    4. 保存到磁盘
    """
```

**存储机制：**
- **FAISS索引**: 使用高效的向量搜索索引
- **内存优化**: 处理大量向量时的内存管理
- **持久化**: 支持保存和加载知识库
- **版本控制**: 知识库的版本管理

#### 5. 搜索阶段 (Search)

```python
# 流程：查询文本 → 向量化 → 相似度搜索 → 结果排序
def search(self, query: str, k: int = 5) -> List[SearchResult]:
    """
    搜索处理流程：
    1. 查询文本向量化
    2. 执行相似度搜索
    3. 结果排序和过滤
    4. 返回最相关结果
    """
```

**搜索算法：**
- **向量化查询**: 将查询转换为向量
- **相似度计算**: 使用余弦相似度或欧氏距离
- **Top-K搜索**: 返回最相似的K个结果
- **结果排序**: 按相似度降序排列

#### 6. 问答生成阶段 (Question Answering)

```python
# 流程：搜索结果 → 上下文构建 → LLM调用 → 答案生成
def ask_question(self, question: str, k: int = 5) -> Dict[str, Any]:
    """
    问答生成流程：
    1. 搜索相关文档片段
    2. 构建上下文提示
    3. 调用LLM生成答案
    4. 返回答案和来源信息
    """
```

**问答机制：**
- **上下文构建**: 将搜索结果组合成上下文
- **提示工程**: 设计有效的提示模板
- **LLM调用**: 使用通义千问或OpenAI生成答案
- **来源追踪**: 记录答案的来源文档和页码

### 组件交互图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF文档输入   │───▶│   文本提取器    │───▶│   文本分块器    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   知识库管理    │◀───│   向量存储器    │◀───│   向量化器      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   搜索引擎      │    │   相似度计算    │    │   API管理器     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   问答生成器    │◀───│   上下文构建器  │◀───│   结果排序器    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 数据流说明

#### 1. 输入数据流
```
PDF文件 → 字节流 → 文本内容 → 文本块列表
```

#### 2. 处理数据流
```
文本块 → 向量表示 → FAISS索引 → 搜索查询 → 相似度结果
```

#### 3. 输出数据流
```
搜索结果 → 上下文 → LLM响应 → 最终答案
```

### 关键算法说明

#### 1. 文本分块算法
```python
def _split_text(self, text: str) -> List[str]:
    """
    智能文本分块算法：
    1. 按分隔符分割（段落、句子、标点）
    2. 控制块大小（默认512字符）
    3. 添加重叠（默认128字符）
    4. 保持语义完整性
    """
```

#### 2. 相似度搜索算法
```python
def _search_similar(self, query_vector: List[float], k: int) -> Tuple[List[int], List[float]]:
    """
    FAISS相似度搜索：
    1. 使用余弦相似度或L2距离
    2. 返回Top-K最相似结果
    3. 包含相似度分数
    """
```

#### 3. 上下文构建算法
```python
def _build_context(self, search_results: List[SearchResult]) -> str:
    """
    上下文构建策略：
    1. 按相似度排序结果
    2. 限制上下文长度
    3. 添加页码信息
    4. 格式化输出
    """
```

### 性能优化策略

#### 1. 内存优化
- **批量处理**: 减少内存占用
- **懒加载**: 按需加载数据
- **垃圾回收**: 及时释放内存

#### 2. API优化
- **批量调用**: 减少API请求次数
- **缓存机制**: 缓存常用结果
- **错误重试**: 提高成功率

#### 3. 搜索优化
- **索引优化**: 使用高效的FAISS索引
- **并行处理**: 多线程搜索
- **结果缓存**: 缓存搜索结果

### 错误处理机制

#### 1. API错误处理
```python
def _handle_api_error(self, error: Exception) -> None:
    """
    错误处理策略：
    1. 网络错误重试
    2. API配额检查
    3. 降级到备用服务
    4. 用户友好提示
    """
```

#### 2. 数据验证
```python
def _validate_input(self, data: Any) -> bool:
    """
    输入验证：
    1. 文件格式检查
    2. 数据完整性验证
    3. 编码格式检查
    """
```

## 最新更新

### v2.1 更新内容
- ✅ 完全修复了LangChain弃用警告问题
- ✅ 集成通义千问（qwen-turbo）替代DeepSeek
- ✅ 解决了API兼容性问题
- ✅ 抑制了FAISS GPU警告
- ✅ 改进了LLM响应处理
- ✅ 添加了更好的错误处理

### v2.0 更新内容
- ✅ 修复了LangChain弃用警告问题
- ✅ 使用最新的LangChain API
- ✅ 改进了问答功能的稳定性
- ✅ 添加了更好的错误处理

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. API密钥配置

系统支持两种API服务，请至少配置一种：

#### 阿里百炼API（推荐）
```bash
# 方式1：使用DASHSCOPE_API_KEY
export DASHSCOPE_API_KEY="your_dashscope_api_key"

# 方式2：使用ALI_API_KEY
export ALI_API_KEY="your_ali_api_key"
```

#### OpenAI API
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### 3. 获取API密钥

#### 阿里百炼API
1. 访问 [阿里百炼官网](https://bailian.console.aliyun.com/)
2. 注册并登录账户
3. 创建应用获取API密钥
4. 开通文本嵌入和对话模型服务

#### OpenAI API
1. 访问 [OpenAI官网](https://platform.openai.com/)
2. 注册并登录账户
3. 在API Keys页面创建密钥
4. 确保账户有足够余额

## 使用方法

### 1. 快速开始

```bash
# 运行设置脚本
python setup.py

# 测试通义千问集成
python test_tongyi.py

# 运行完整测试
python main.py

# 运行示例
python example.py
```

### 2. 基本使用

```python
from main import RAGSystem, RAGConfig

# 创建配置
config = RAGConfig(
    chunk_size=512,
    chunk_overlap=128,
    llm_model="qwen-turbo",  # 使用通义千问
    save_path="./knowledge_base"
)

# 创建RAG系统
rag_system = RAGSystem(config)

# 处理PDF文件
knowledge_base = rag_system.process_pdf("your_pdf_file.pdf")

# 搜索相关内容
results = rag_system.search("你的问题", k=5)

# 智能问答
answer_info = rag_system.ask_question("你的问题", k=5)
```

## 功能特性

### 1. 智能API选择
- 优先使用阿里百炼API
- 自动降级到OpenAI API
- 支持多种API密钥环境变量

### 2. 错误处理
- API配额超限检测
- 网络错误处理
- 用户友好的错误信息

### 3. 知识库管理
- 自动保存和加载
- 页码信息记录
- 持久化存储

### 4. 搜索功能
- 相似度搜索
- 多结果返回
- 页码追踪

### 5. 现代化架构
- 使用最新的LangChain API
- 无弃用警告
- 更好的稳定性

### 6. 通义千问集成
- 使用qwen-turbo模型
- 更好的中文支持
- 稳定的API调用

## 故障排除

### 常见问题

#### 1. API配额超限
```
错误: You exceeded your current quota
解决: 检查账户余额或等待配额重置
```

#### 2. API密钥无效
```
错误: 401 Unauthorized
解决: 检查API密钥是否正确配置
```

#### 3. 网络连接问题
```
错误: Connection timeout
解决: 检查网络连接或使用代理
```

#### 4. LangChain警告（已修复）
```
警告: LangChainDeprecationWarning
解决: 已使用最新API，不再出现此警告
```

#### 5. FAISS GPU警告（已修复）
```
警告: Failed to load GPU Faiss
解决: 已抑制此警告，不影响功能
```

### 调试建议

1. 查看日志文件 `rag_system.log`
2. 检查环境变量配置
3. 确认API服务状态
4. 验证账户余额
5. 运行 `python test_tongyi.py` 验证通义千问集成

## 配置参数

### RAGConfig 参数说明

- `chunk_size`: 文本块大小（默认512）
- `chunk_overlap`: 文本块重叠大小（默认128）
- `separators`: 文本分割分隔符
- `embedding_model`: 嵌入模型名称
- `llm_model`: 大语言模型名称（默认qwen-turbo）
- `save_path`: 知识库保存路径

## 性能优化

### 1. 文本分块优化
- 根据文档类型调整 `chunk_size`
- 设置合适的 `chunk_overlap`
- 优化分隔符配置

### 2. API使用优化
- 批量处理减少API调用
- 使用本地缓存
- 合理设置搜索数量

## 扩展功能

### 1. 支持更多文档格式
- Word文档 (.docx)
- 文本文件 (.txt)
- Markdown文件 (.md)

### 2. 支持更多向量数据库
- ChromaDB
- Pinecone
- Weaviate

### 3. 支持更多LLM
- 本地模型
- 其他云服务API

## 许可证

MIT License