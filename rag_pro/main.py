"""
RAG Pro 系统 - 高级检索增强生成系统
使用MultiQueryRetriever进行多查询检索，提高检索准确性

主要功能:
- 多查询检索：自动生成多个相关查询
- 向量相似度搜索：基于语义的文档检索
- 智能问答：集成大语言模型
- 错误处理和日志记录
"""

import os
import logging
import warnings
from typing import List, Dict, Any
from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.schema import Document

# 抑制FAISS GPU警告
warnings.filterwarnings("ignore", message=".*Failed to load GPU Faiss.*")
warnings.filterwarnings("ignore", message=".*GpuIndexIVFFlat.*")

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_pro.log', encoding='utf-8')
        ]
    )

def check_api_key():
    """检查API密钥配置"""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALI_API_KEY")
    if not api_key:
        raise ValueError("未配置API密钥，请设置DASHSCOPE_API_KEY或ALI_API_KEY环境变量")
    return api_key

def initialize_models(api_key: str):
    """
    初始化模型
    
    Args:
        api_key: 阿里百炼API密钥
    
    Returns:
        tuple: (llm, embeddings) 大语言模型和嵌入模型
    """
    try:
        # 初始化大语言模型 - 使用通义千问
        llm = Tongyi(
            model_name="qwen-turbo",  # 使用turbo版本，更稳定
            dashscope_api_key=api_key
        )
        
        # 创建嵌入模型 - 使用v2版本，与知识库兼容
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",  # 使用v2版本确保兼容性
            dashscope_api_key=api_key
        )
        
        logging.info("模型初始化成功")
        return llm, embeddings
        
    except Exception as e:
        logging.error(f"模型初始化失败: {e}")
        raise

def load_vectorstore(embeddings, knowledge_base_path: str = "./knowledge_base"):
    """
    加载向量数据库
    
    Args:
        embeddings: 嵌入模型
        knowledge_base_path: 知识库路径
    
    Returns:
        FAISS: 向量数据库对象
    """
    try:
        if not os.path.exists(knowledge_base_path):
            raise FileNotFoundError(f"知识库路径不存在: {knowledge_base_path}")
        
        # 加载向量数据库，添加安全参数
        vectorstore = FAISS.load_local(
            knowledge_base_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        logging.info(f"向量数据库加载成功，包含 {len(vectorstore.index_to_docstore_id)} 个文档")
        return vectorstore
        
    except Exception as e:
        logging.error(f"向量数据库加载失败: {e}")
        raise

def create_retriever(vectorstore, llm):
    """
    创建多查询检索器
    
    Args:
        vectorstore: 向量数据库
        llm: 大语言模型
    
    Returns:
        MultiQueryRetriever: 多查询检索器
    """
    try:
        # 创建基础检索器
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # 返回前5个最相关文档
        )
        
        # 创建多查询检索器
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            parser_key="text"  # 使用文本解析器
        )
        
        logging.info("多查询检索器创建成功")
        return retriever
        
    except Exception as e:
        logging.error(f"检索器创建失败: {e}")
        raise

def search_documents(retriever, query: str) -> List[Document]:
    """
    搜索相关文档
    
    Args:
        retriever: 检索器
        query: 查询文本
    
    Returns:
        List[Document]: 相关文档列表
    """
    try:
        # 使用新的invoke方法替代已弃用的get_relevant_documents
        results = retriever.invoke(query)
        
        logging.info(f"查询 '{query}' 找到 {len(results)} 个相关文档")
        return results
        
    except Exception as e:
        logging.error(f"文档搜索失败: {e}")
        raise

def display_results(query: str, results: List[Document]):
    """
    显示搜索结果
    
    Args:
        query: 原始查询
        results: 搜索结果列表
    """
    print("\n" + "=" * 60)
    print("搜索结果")
    print("=" * 60)
    print(f"查询: {query}")
    print(f"找到 {len(results)} 个相关文档:")
    
    for i, doc in enumerate(results, 1):
        print(f"\n文档 {i}:")
        # 显示文档内容的前200个字符
        content = doc.page_content
        if len(content) > 200:
            print(content[:200] + "...")
        else:
            print(content)
        
        # 显示元数据（如果有）
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"元数据: {doc.metadata}")

def main():
    """
    主函数 - RAG Pro系统的主要执行流程
    """
    print("=" * 60)
    print("RAG Pro 系统启动")
    print("=" * 60)
    
    try:
        # 1. 设置日志系统
        setup_logging()
        logging.info("RAG Pro系统启动")
        
        # 2. 检查API密钥
        api_key = check_api_key()
        logging.info("API密钥检查通过")
        
        # 3. 初始化模型
        llm, embeddings = initialize_models(api_key)
        
        # 4. 加载向量数据库
        vectorstore = load_vectorstore(embeddings)
        
        # 5. 创建检索器
        retriever = create_retriever(vectorstore, llm)
        
        # 6. 执行查询
        test_queries = [
            "客户经理的考核标准是什么？",
            "客户经理每年评聘申报时间是怎样的？",
            "客户经理被投诉了，投诉一次扣多少分？"
        ]
        
        for query in test_queries:
            try:
                # 搜索相关文档
                results = search_documents(retriever, query)
                
                # 显示结果
                display_results(query, results)
                
            except Exception as e:
                logging.error(f"查询 '{query}' 失败: {e}")
                print(f"❌ 查询失败: {e}")
        
        print("\n" + "=" * 60)
        print("RAG Pro 系统执行完成")
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"系统执行失败: {e}")
        print(f"❌ 系统执行失败: {e}")
        print("\n💡 故障排除建议:")
        print("  1. 检查API密钥是否正确配置")
        print("  2. 确认知识库路径是否存在")
        print("  3. 检查网络连接")
        print("  4. 查看日志文件获取详细错误信息")

if __name__ == "__main__":
    main()
