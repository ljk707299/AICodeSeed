# 安装和使用指南

## 🛠️ 环境准备

### 1. 系统要求

- Python 3.8+
- macOS/Linux/Windows
- 至少4GB内存（推荐8GB+）

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 安装依赖包

```bash
# 基础依赖
pip install numpy openai

# 向量数据库依赖
pip install chromadb faiss-cpu pinecone-client

# 升级pip（可选）
pip install --upgrade pip
```

### 4. 验证安装

```bash
# 检查Python版本
python --version

# 检查已安装的包
pip list | grep -E "(chroma|faiss|pinecone|openai)"
```

## 🔑 API密钥配置

### 阿里百炼API（可选）

```bash
# 设置环境变量
export ALI_API_KEY="your_ali_api_key"
export ALI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 或在代码中直接设置
```

### OpenAI API（可选）

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### Pinecone API（可选）

```bash
export PINECONE_API_KEY="your_pinecone_api_key"
```

## 🚀 运行示例

### 1. 阿里百炼嵌入向量示例

```bash
python main.py
```

**预期输出**:
```
=== 基本功能测试 ===
向量维度: 1536
前10个元素: [-0.123, 0.456, ...]

=== 相似度计算示例 ===
余弦相似度结果:
文档1: 0.8234 - 联合国就苏丹达尔富尔地区大规模暴力事件发出警告
...
```

### 2. ChromaDB示例

```bash
python chroma_main.py
```

**预期输出**:
```
=== ChromaDB 向量数据库演示 ===

1. 创建/获取集合...
创建新的集合

2. 添加文档到集合...
成功添加 5 个文档到集合中

=== 文档查询示例 ===
1. 基于文本的相似度搜索:
查询结果:
  结果1: 距离=0.4612, 来源=RAG
      文档: RAG是一种检索增强生成技术，结合了检索和生成两种能力
...
```

### 3. FAISS示例

```bash
python faiss_example.py
```

**预期输出**:
```
=== FAISS 向量数据库演示 ===

创建 1000 个 128 维向量...

1. Flat索引演示（精确搜索）:
成功添加 1000 个向量到索引中
搜索结果:
  结果1: 索引=0, 相似度=1.0000
  结果2: 索引=123, 相似度=0.8234
...
```

### 4. Pinecone示例

```bash
python pinecone_example.py
```

**预期输出**:
```
=== Pinecone 向量数据库演示 ===

未设置 PINECONE_API_KEY 环境变量
继续演示概念和模拟操作...

=== 嵌入向量生成演示 ===
正在生成模拟嵌入向量...
成功生成 5 个嵌入向量
每个向量维度: 1536
...
```

## 📊 性能测试

### 运行性能对比

```bash
# 创建性能测试脚本
cat > performance_test.py << 'EOF'
import time
import numpy as np
from chroma_main import create_collection_with_config, add_documents_to_collection
from faiss_example import create_sample_vectors, create_flat_index, add_vectors_to_index

def test_chromadb_performance():
    print("测试ChromaDB性能...")
    start_time = time.time()
    
    collection = create_collection_with_config()
    documents = [f"文档{i}" for i in range(1000)]
    add_documents_to_collection(collection)
    
    end_time = time.time()
    print(f"ChromaDB插入1000个文档耗时: {end_time - start_time:.2f}秒")

def test_faiss_performance():
    print("测试FAISS性能...")
    start_time = time.time()
    
    vectors = create_sample_vectors(128, 1000)
    index = create_flat_index(128)
    add_vectors_to_index(index, vectors)
    
    end_time = time.time()
    print(f"FAISS插入1000个向量耗时: {end_time - start_time:.2f}秒")

if __name__ == "__main__":
    test_chromadb_performance()
    test_faiss_performance()
EOF

# 运行性能测试
python performance_test.py
```

## 🔧 故障排除

### 常见错误及解决方案

#### 1. FAISS安装错误

**错误**: `ERROR: Could not find a version that satisfies the requirement faiss`

**解决方案**:
```bash
# 对于Apple Silicon Mac
pip install faiss-cpu

# 对于Intel Mac/Linux
pip install faiss-cpu

# 如果仍有问题，尝试从conda安装
conda install -c conda-forge faiss-cpu
```

#### 2. ChromaDB启动错误

**错误**: `Failed to send telemetry event`

**解决方案**:
```bash
# 这是警告，不影响功能，可以忽略
# 或者设置环境变量禁用遥测
export CHROMA_TELEMETRY_ENABLED=false
```

#### 3. Pinecone API错误

**错误**: `AttributeError: module 'pinecone' has no attribute 'Index'`

**解决方案**:
```bash
# 更新到最新版本
pip install --upgrade pinecone-client

# 或者使用兼容的API版本
pip install pinecone-client==2.2.4
```

#### 4. 内存不足错误

**错误**: `MemoryError` 或 `OutOfMemoryError`

**解决方案**:
```bash
# 减少向量数量或维度
# 在代码中修改：
# dimension = 64  # 从128减少到64
# num_vectors = 100  # 从1000减少到100
```

### 调试技巧

1. **检查环境变量**
```bash
echo $ALI_API_KEY
echo $OPENAI_API_KEY
echo $PINECONE_API_KEY
```

2. **检查Python路径**
```bash
which python
python -c "import sys; print(sys.path)"
```

3. **检查包版本**
```bash
pip show chromadb
pip show faiss-cpu
pip show pinecone-client
```

## 📈 扩展使用

### 自定义配置

1. **修改向量维度**
```python
# 在main.py中
dimension = 768  # 从1536改为768

# 在faiss_example.py中
dimension = 256  # 从128改为256
```

2. **调整搜索参数**
```python
# ChromaDB
n_results = 10  # 从3改为10

# FAISS
k = 20  # 从5改为20
```

3. **添加更多文档**
```python
# 扩展文档列表
documents = [
    "文档1",
    "文档2",
    # ... 添加更多文档
    "文档100"
]
```

### 集成到其他项目

1. **作为模块导入**
```python
from main import get_embeddings, cos_sim
from chroma_main import create_collection_with_config
from faiss_example import create_flat_index
```

2. **批量处理**
```python
# 批量处理文档
def process_documents(doc_list):
    vectors = get_embeddings(doc_list)
    # 进一步处理...
    return vectors
```

## 📞 获取帮助

如果遇到问题：

1. 检查本指南的故障排除部分
2. 查看各项目的官方文档
3. 在GitHub上提交Issue
4. 联系技术支持

## 🎯 下一步

完成基础示例后，可以：

1. 尝试不同的索引类型
2. 优化性能参数
3. 集成到实际项目中
4. 学习高级功能 