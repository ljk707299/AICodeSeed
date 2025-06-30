"""
RAG (Retrieval-Augmented Generation) 系统
基于PDF文档的问答系统，支持文本提取、向量化和相似度搜索

主要功能：
1. PDF文档文本提取和页码记录
2. 文本分块处理
3. 向量化存储（支持FAISS）
4. 相似度搜索
5. 知识库持久化
6. 智能问答（支持多种LLM）

支持的嵌入模型：
- 阿里百炼 (DashScope)
- OpenAI

支持的大语言模型：
- 阿里百炼 (通义千问)
- OpenAI GPT

作者: AI助手
版本: v2.1
更新日期: 2024-06-29
"""

import os
import logging
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# 抑制FAISS GPU警告 - 这些警告不影响功能，只是提示GPU功能不可用
warnings.filterwarnings("ignore", message=".*Failed to load GPU Faiss.*")
warnings.filterwarnings("ignore", message=".*GpuIndexIVFFlat.*")

# 尝试加载.env文件 - 用于环境变量配置
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装python-dotenv，跳过.env文件加载
    pass

# LangChain相关导入 - 核心功能库
from PyPDF2 import PdfReader  # PDF文件读取
from langchain_openai import OpenAI, ChatOpenAI  # OpenAI模型接口
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里百炼嵌入模型
from langchain_community.callbacks.manager import get_openai_callback  # API成本跟踪
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_community.vectorstores import FAISS  # 向量数据库
from langchain_community.llms import Tongyi  # 通义千问模型
# 使用新的提示模板API - 替代已弃用的问答链
from langchain_core.prompts import PromptTemplate


@dataclass
class RAGConfig:
    """
    RAG系统配置类
    用于统一管理系统的各种参数和设置
    
    属性说明:
        chunk_size: 文本块大小，影响向量化的粒度
        chunk_overlap: 文本块重叠大小，确保上下文连续性
        separators: 文本分割分隔符，按优先级排列
        embedding_model: 嵌入模型名称
        llm_model: 大语言模型名称
        save_path: 知识库保存路径
    """
    chunk_size: int = 512          # 文本块大小 - 建议512-1024之间
    chunk_overlap: int = 128       # 文本块重叠大小 - 通常为chunk_size的1/4
    separators: List[str] = None   # 文本分割分隔符
    embedding_model: str = "text-embedding-v2"  # 阿里百炼嵌入模型
    llm_model: str = "qwen-turbo"  # 通义千问模型 - 免费且中文支持好
    save_path: Optional[str] = None  # 知识库保存路径
    
    def __post_init__(self):
        """初始化后处理，设置默认分隔符"""
        if self.separators is None:
            # 按优先级设置分隔符：段落 -> 句子 -> 单词 -> 字符
            # 这样可以保持语义的完整性
            self.separators = ["\n\n", "\n", ".", " ", ""]


class PDFProcessor:
    """
    PDF文档处理器
    负责从PDF文件中提取文本内容并记录页码信息
    
    主要功能:
        - PDF文件读取和验证
        - 文本提取和清理
        - 页码信息记录
        - 错误处理和日志记录
    """
    
    def __init__(self, config: RAGConfig):
        """
        初始化PDF处理器
        
        Args:
            config: RAG系统配置对象，包含处理参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)  # 获取当前模块的日志器
    
    def extract_text_with_page_numbers(self, pdf_path: str) -> Tuple[str, List[int]]:
        """
        从PDF中提取文本并记录每行文本对应的页码
        
        处理流程:
        1. 验证PDF文件存在性
        2. 逐页读取PDF内容
        3. 提取文本并记录页码
        4. 错误处理和日志记录
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            Tuple[str, List[int]]: (提取的文本内容, 每行文本对应的页码列表)
        
        Raises:
            FileNotFoundError: PDF文件不存在
            ValueError: PDF文件中未提取到任何文本内容
            Exception: PDF读取失败
        """
        try:
            # 使用pathlib处理路径，更加安全和跨平台
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
            
            # 读取PDF文件 - 使用str()确保路径格式正确
            pdf_reader = PdfReader(str(pdf_path))
            
            text = ""  # 存储提取的文本
            page_numbers = []  # 存储每行文本对应的页码
            
            # 逐页处理PDF内容
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                try:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        # 添加换行符确保文本格式正确
                        text += extracted_text + "\n"
                        # 为每行文本记录对应的页码
                        lines = extracted_text.split("\n")
                        page_numbers.extend([page_number] * len(lines))
                    else:
                        self.logger.warning(f"第 {page_number} 页未找到文本内容")
                except Exception as e:
                    # 单页处理失败不影响整体处理
                    self.logger.error(f"处理第 {page_number} 页时出错: {e}")
                    continue
            
            # 检查是否成功提取到文本
            if not text.strip():
                raise ValueError("PDF文件中未提取到任何文本内容")
            
            self.logger.info(f"成功从PDF提取文本，共 {len(pdf_reader.pages)} 页")
            return text.strip(), page_numbers
            
        except Exception as e:
            self.logger.error(f"PDF处理失败: {e}")
            raise


class TextProcessor:
    """
    文本处理器
    负责将长文本分割成适合向量化的小块
    
    主要功能:
        - 文本分割策略配置
        - 智能文本分块
        - 保持语义完整性
    """
    
    def __init__(self, config: RAGConfig):
        """
        初始化文本处理器
        
        Args:
            config: RAG系统配置对象，包含分割参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        创建文本分割器
        
        分割策略:
        - 优先按段落分割
        - 其次按句子分割
        - 最后按单词分割
        - 确保每个块不超过指定大小
        
        Returns:
            RecursiveCharacterTextSplitter: 配置好的文本分割器
        """
        return RecursiveCharacterTextSplitter(
            separators=self.config.separators,      # 分割分隔符，按优先级排列
            chunk_size=self.config.chunk_size,      # 块大小，控制向量化粒度
            chunk_overlap=self.config.chunk_overlap, # 重叠大小，保持上下文连续性
            length_function=len,                    # 长度计算函数
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本为小块
        
        处理流程:
        1. 创建文本分割器
        2. 执行文本分割
        3. 记录分割结果
        
        Args:
            text: 要分割的文本
        
        Returns:
            List[str]: 分割后的文本块列表
        """
        text_splitter = self.create_text_splitter()
        chunks = text_splitter.split_text(text)
        self.logger.info(f"文本被分割成 {len(chunks)} 个块")
        return chunks


class EmbeddingManager:
    """
    嵌入向量管理器
    负责管理不同的嵌入模型，支持多种嵌入服务
    
    主要功能:
        - 嵌入模型初始化
        - API密钥管理
        - 模型选择策略
        - 懒加载优化
    """
    
    def __init__(self, config: RAGConfig):
        """
        初始化嵌入管理器
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._embeddings = None  # 懒加载嵌入模型 - 只在需要时初始化
    
    def get_embeddings(self):
        """
        获取嵌入模型实例（懒加载模式）
        
        选择策略:
        1. 优先使用阿里百炼（免费额度大）
        2. 备选使用OpenAI
        3. 检查API密钥配置
        
        Returns:
            嵌入模型实例
            
        Raises:
            ValueError: 未配置API密钥
        """
        if self._embeddings is None:
            try:
                # 优先使用阿里百炼，检查多个可能的API密钥环境变量
                dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALI_API_KEY")
                if dashscope_key:
                    self._embeddings = DashScopeEmbeddings(
                        model=self.config.embedding_model,
                        dashscope_api_key=dashscope_key
                    )
                    self.logger.info("使用阿里百炼嵌入模型")
                elif os.getenv("OPENAI_API_KEY"):
                    self._embeddings = OpenAIEmbeddings()
                    self.logger.info("使用OpenAI嵌入模型")
                else:
                    raise ValueError("未配置嵌入模型API密钥，请设置DASHSCOPE_API_KEY或OPENAI_API_KEY")
            except Exception as e:
                self.logger.error(f"初始化嵌入模型失败: {e}")
                raise
        
        return self._embeddings


class LLMManager:
    """
    大语言模型管理器
    负责管理不同的LLM，支持多种对话模型
    
    主要功能:
        - LLM模型初始化
        - API密钥管理
        - 模型选择策略
        - 响应格式处理
    """
    
    def __init__(self, config: RAGConfig):
        """
        初始化LLM管理器
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._llm = None  # 懒加载LLM - 只在需要时初始化
    
    def get_llm(self):
        """
        获取大语言模型实例（懒加载模式）
        
        选择策略:
        1. 优先使用通义千问（中文支持好，免费额度大）
        2. 备选使用OpenAI GPT
        3. 检查API密钥配置
        
        Returns:
            大语言模型实例
            
        Raises:
            ValueError: 未配置API密钥
        """
        if self._llm is None:
            try:
                # 优先使用阿里百炼通义千问
                dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALI_API_KEY")
                if dashscope_key:
                    self._llm = Tongyi(
                        model_name=self.config.llm_model,
                        dashscope_api_key=dashscope_key
                    )
                    self.logger.info(f"使用通义千问LLM: {self.config.llm_model}")
                elif os.getenv("OPENAI_API_KEY"):
                    self._llm = ChatOpenAI(
                        model="gpt-3.5-turbo"
                    )
                    self.logger.info("使用OpenAI GPT模型")
                else:
                    raise ValueError("未配置LLM API密钥，请设置DASHSCOPE_API_KEY或OPENAI_API_KEY")
            except Exception as e:
                self.logger.error(f"初始化LLM失败: {e}")
                raise
        
        return self._llm


class VectorStoreManager:
    """
    向量存储管理器
    负责创建、保存和加载向量数据库
    
    主要功能:
        - FAISS向量数据库管理
        - 知识库持久化
        - 页码信息存储
        - 错误处理和恢复
    """
    
    def __init__(self, config: RAGConfig):
        """
        初始化向量存储管理器
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.embedding_manager = EmbeddingManager(config)  # 嵌入管理器
    
    def create_knowledge_base(self, chunks: List[str], page_numbers: List[int]) -> FAISS:
        """
        从文本块创建知识库
        
        处理流程:
        1. 获取嵌入模型
        2. 创建FAISS向量数据库
        3. 存储页码信息
        4. 错误处理和日志记录
        
        Args:
            chunks: 文本块列表
            page_numbers: 页码信息列表
        
        Returns:
            FAISS: 向量存储对象
        """
        try:
            # 获取嵌入模型
            embeddings = self.embedding_manager.get_embeddings()
            
            # 创建FAISS向量数据库
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            # 存储每个文本块对应的页码信息
            page_info = {chunk: page_numbers[i] for i, chunk in enumerate(chunks)}
            knowledge_base.page_info = page_info
            
            self.logger.info("知识库创建成功")
            return knowledge_base
            
        except Exception as e:
            self.logger.error(f"创建知识库失败: {e}")
            raise
    
    def save_knowledge_base(self, knowledge_base: FAISS, save_path: str):
        """
        保存知识库到本地
        
        保存内容:
        - FAISS向量数据库文件
        - 页码信息pickle文件
        
        Args:
            knowledge_base: 要保存的知识库
            save_path: 保存路径
        """
        try:
            save_path = Path(save_path)
            # 确保目录存在
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存FAISS向量数据库
            knowledge_base.save_local(str(save_path))
            
            # 保存页码信息到pickle文件
            page_info_path = save_path / "page_info.pkl"
            with open(page_info_path, "wb") as f:
                pickle.dump(knowledge_base.page_info, f)
            
            self.logger.info(f"知识库已保存到: {save_path}")
            
        except Exception as e:
            self.logger.error(f"保存知识库失败: {e}")
            raise
    
    def load_knowledge_base(self, load_path: str) -> FAISS:
        """
        从本地加载知识库
        
        加载内容:
        - FAISS向量数据库文件
        - 页码信息pickle文件
        
        Args:
            load_path: 加载路径
        
        Returns:
            FAISS: 加载的知识库对象
            
        Raises:
            FileNotFoundError: 知识库路径不存在
        """
        try:
            load_path = Path(load_path)
            if not load_path.exists():
                raise FileNotFoundError(f"知识库路径不存在: {load_path}")
            
            # 获取嵌入模型
            embeddings = self.embedding_manager.get_embeddings()
            
            # 加载FAISS向量数据库
            knowledge_base = FAISS.load_local(
                str(load_path), 
                embeddings, 
                allow_dangerous_deserialization=True  # 允许反序列化
            )
            
            # 加载页码信息
            page_info_path = load_path / "page_info.pkl"
            if page_info_path.exists():
                with open(page_info_path, "rb") as f:
                    page_info = pickle.load(f)
                knowledge_base.page_info = page_info
                self.logger.info("页码信息已加载")
            else:
                self.logger.warning("未找到页码信息文件")
            
            self.logger.info(f"知识库已从 {load_path} 加载")
            return knowledge_base
            
        except Exception as e:
            self.logger.error(f"加载知识库失败: {e}")
            raise


class RAGSystem:
    """
    RAG系统主类
    整合所有组件，提供完整的RAG功能
    
    主要功能:
        - PDF文档处理
        - 知识库创建和管理
        - 相似度搜索
        - 智能问答
        - 错误处理和日志记录
    """
    
    def __init__(self, config: RAGConfig):
        """
        初始化RAG系统
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个功能组件
        self.pdf_processor = PDFProcessor(config)           # PDF处理器
        self.text_processor = TextProcessor(config)         # 文本处理器
        self.vector_store_manager = VectorStoreManager(config)  # 向量存储管理器
        self.llm_manager = LLMManager(config)               # LLM管理器
        
        # 知识库实例（初始为空）
        self.knowledge_base: Optional[FAISS] = None
    
    def process_pdf(self, pdf_path: str, save_path: Optional[str] = None) -> FAISS:
        """
        处理PDF文件并创建知识库
        
        处理流程:
        1. 提取PDF文本和页码信息
        2. 分割文本为小块
        3. 创建向量知识库
        4. 保存知识库（如果指定了保存路径）
        
        Args:
            pdf_path: PDF文件路径
            save_path: 可选，保存路径
        
        Returns:
            FAISS: 创建的知识库
        """
        try:
            # 步骤1: 提取PDF文本和页码信息
            text, page_numbers = self.pdf_processor.extract_text_with_page_numbers(pdf_path)
            
            # 步骤2: 分割文本为小块
            chunks = self.text_processor.split_text(text)
            
            # 步骤3: 创建向量知识库
            self.knowledge_base = self.vector_store_manager.create_knowledge_base(chunks, page_numbers)
            
            # 步骤4: 保存知识库（如果指定了保存路径）
            if save_path or self.config.save_path:
                save_path = save_path or self.config.save_path
                self.vector_store_manager.save_knowledge_base(self.knowledge_base, save_path)
            
            return self.knowledge_base
            
        except Exception as e:
            self.logger.error(f"处理PDF失败: {e}")
            raise
    
    def load_knowledge_base(self, load_path: str) -> FAISS:
        """
        加载已存在的知识库
        
        Args:
            load_path: 知识库路径
        
        Returns:
            FAISS: 加载的知识库
        """
        self.knowledge_base = self.vector_store_manager.load_knowledge_base(load_path)
        return self.knowledge_base
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, int]]:
        """
        在知识库中搜索相关内容
        
        搜索流程:
        1. 验证知识库是否已初始化
        2. 执行相似度搜索
        3. 整理搜索结果，包含页码信息
        
        Args:
            query: 查询文本 - 用户想要搜索的问题或关键词
            k: 返回结果数量 - 默认返回5个最相关的结果
        
        Returns:
            List[Tuple[str, float, int]]: 搜索结果列表
                - str: 文本内容
                - float: 相似度分数（越小越相似）
                - int: 页码信息
        
        Raises:
            ValueError: 知识库未初始化
        """
        if self.knowledge_base is None:
            raise ValueError("知识库未初始化，请先处理PDF或加载知识库")
        
        try:
            # 执行相似度搜索 - 使用FAISS向量数据库
            docs_and_scores = self.knowledge_base.similarity_search_with_score(query, k=k)
            
            # 整理搜索结果，包含页码信息
            results = []
            for doc, score in docs_and_scores:
                # 从知识库的页码信息中获取对应的页码
                page_number = self.knowledge_base.page_info.get(doc.page_content, 0)
                results.append((doc.page_content, score, page_number))
            
            return results
            
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            raise
    
    def ask_question(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        使用RAG系统回答问题
        
        问答流程:
        1. 验证知识库是否已初始化
        2. 搜索相关文档
        3. 构建提示模板
        4. 调用LLM生成答案
        5. 整理答案和来源信息
        
        Args:
            query: 问题 - 用户想要了解的问题
            k: 检索文档数量 - 用于生成答案的相关文档数量
        
        Returns:
            Dict[str, Any]: 包含答案和来源信息的字典
                - answer: 生成的答案
                - sources: 来源页码列表
                - cost: API调用成本
                - docs_used: 使用的文档数量
        """
        if self.knowledge_base is None:
            raise ValueError("知识库未初始化，请先处理PDF或加载知识库")
        
        try:
            # 获取LLM实例 - 通义千问或OpenAI
            llm = self.llm_manager.get_llm()
            
            # 搜索相关文档 - 找到与问题最相关的文档片段
            docs = self.knowledge_base.similarity_search(query, k=k)
            
            # 构建上下文 - 将所有相关文档内容合并
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 创建提示模板 - 指导LLM如何基于上下文回答问题
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的信息中找到答案。

上下文信息：
{context}

问题：{question}

请提供准确、详细的答案："""
            )
            
            # 构建完整提示 - 将上下文和问题填入模板
            full_prompt = prompt_template.format(context=context, question=query)
            
            # 使用回调函数跟踪API调用成本
            with get_openai_callback() as cost:
                # 直接调用LLM - 发送完整提示给模型
                response = llm.invoke(full_prompt)
                
                # 处理不同LLM的响应格式 - 兼容不同的模型输出格式
                if hasattr(response, 'content'):
                    answer = response.content  # ChatOpenAI格式
                elif hasattr(response, 'text'):
                    answer = response.text     # 其他格式
                elif isinstance(response, str):
                    answer = response          # 字符串格式
                else:
                    answer = str(response)     # 其他格式转换为字符串
                
                # 记录唯一的页码 - 避免重复显示相同页码
                unique_pages = set()
                for doc in docs:
                    text_content = getattr(doc, "page_content", "")
                    source_page = self.knowledge_base.page_info.get(text_content.strip(), "未知")
                    if source_page not in unique_pages:
                        unique_pages.add(source_page)
                
                # 返回完整的答案信息
                return {
                    "answer": answer,                    # 生成的答案
                    "sources": list(unique_pages),       # 来源页码列表
                    "cost": str(cost),                   # API调用成本
                    "docs_used": len(docs)               # 使用的文档数量
                }
                
        except Exception as e:
            self.logger.error(f"问答失败: {e}")
            raise


def setup_logging(level: str = "INFO"):
    """
    设置日志配置
    
    配置内容:
    - 日志级别：控制输出详细程度
    - 日志格式：时间戳、模块名、级别、消息
    - 输出目标：同时输出到控制台和文件
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            - DEBUG: 最详细，包含所有信息
            - INFO: 一般信息，适合生产环境
            - WARNING: 警告信息
            - ERROR: 错误信息
            - CRITICAL: 严重错误
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler('rag_system.log', encoding='utf-8')  # 输出到文件
        ]
    )


def check_api_keys():
    """
    检查API密钥配置
    
    检查内容:
    - 阿里百炼API密钥（DASHSCOPE_API_KEY 或 ALI_API_KEY）
    - OpenAI API密钥（OPENAI_API_KEY）
    - 配置状态和可用性
    
    Returns:
        Dict[str, bool]: 各API密钥的配置状态
            - dashscope_configured: 阿里百炼API是否已配置
            - openai_configured: OpenAI API是否已配置
            - any_configured: 是否有任何API已配置
    """
    # 检查多种可能的API密钥环境变量名称
    dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # 构建配置状态字典
    status = {
        "dashscope_configured": bool(dashscope_key),
        "openai_configured": bool(openai_key),
        "any_configured": bool(dashscope_key or openai_key)
    }
    
    # 显示配置检查结果
    print("API密钥配置检查:")
    print(f"  阿里百炼API: {'✅ 已配置' if status['dashscope_configured'] else '❌ 未配置'}")
    print(f"  OpenAI API: {'✅ 已配置' if status['openai_configured'] else '❌ 未配置'}")
    
    # 如果没有配置任何API密钥，提供帮助信息
    if not status['any_configured']:
        print("\n⚠️  警告: 未配置任何API密钥!")
        print("请设置以下环境变量之一:")
        print("  - DASHSCOPE_API_KEY 或 ALI_API_KEY (推荐)")
        print("  - OPENAI_API_KEY")
    
    return status


def handle_api_error(error: Exception) -> str:
    """
    处理API错误并提供用户友好的错误信息
    
    错误类型识别:
    - 配额超限 (429): API使用量超出限制
    - 认证失败 (401): API密钥无效或过期
    - 权限不足 (403): 账户权限不足
    - 网络超时: 网络连接问题
    - 其他错误: 未知错误类型
    
    Args:
        error: 异常对象 - 包含原始错误信息
    
    Returns:
        str: 用户友好的错误信息 - 便于理解和解决
    """
    error_str = str(error)
    
    # 根据错误信息特征判断错误类型
    if "429" in error_str or "quota" in error_str.lower():
        return "API配额已用完，请检查您的账户余额或等待配额重置"
    elif "401" in error_str or "unauthorized" in error_str.lower():
        return "API密钥无效，请检查您的API密钥配置"
    elif "403" in error_str or "forbidden" in error_str.lower():
        return "API访问被拒绝，请检查您的账户权限"
    elif "timeout" in error_str.lower():
        return "API请求超时，请检查网络连接"
    else:
        return f"API调用失败: {error_str}"


def test_rag_system():
    """
    测试RAG系统功能
    包含完整的PDF处理、知识库创建、搜索和问答测试
    
    测试内容:
    1. PDF文件处理测试 - 验证PDF文本提取功能
    2. 相似度搜索测试 - 验证向量搜索功能
    3. 智能问答测试 - 验证LLM问答功能
    4. 知识库持久化测试 - 验证保存和加载功能
    
    测试流程:
    1. 环境检查和配置
    2. 系统初始化
    3. 功能测试执行
    4. 结果验证和报告
    """
    print("=" * 60)
    print("开始RAG系统测试")
    print("=" * 60)
    
    # 设置日志系统 - 记录测试过程中的详细信息
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 检查API密钥配置 - 确保有可用的API服务
    api_status = check_api_keys()
    if not api_status['any_configured']:
        print("\n❌ 无法继续测试：未配置API密钥")
        return
    
    # 创建RAG系统配置 - 设置测试参数
    config = RAGConfig(
        chunk_size=512,        # 文本块大小 - 适合大多数文档
        chunk_overlap=128,     # 重叠大小 - 保持上下文连续性
        save_path="./knowledge_base"  # 知识库保存路径
    )
    
    # 创建RAG系统实例 - 初始化所有组件
    rag_system = RAGSystem(config)
    
    try:
        # 测试1: PDF文件处理
        logger.info("1. 测试PDF文件处理...")
        pdf_path = './浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf'
        
        if os.path.exists(pdf_path):
            logger.info(f"找到PDF文件: {pdf_path}")
            try:
                # 处理PDF并创建知识库
                knowledge_base = rag_system.process_pdf(pdf_path)
                logger.info("✅ PDF处理完成！")
                logger.info(f"知识库包含 {len(knowledge_base.page_info)} 个文本块")
            except Exception as e:
                error_msg = handle_api_error(e)
                logger.error(f"❌ PDF处理失败: {error_msg}")
                print(f"❌ PDF处理失败: {error_msg}")
                return
        else:
            logger.error(f"❌ PDF文件不存在: {pdf_path}")
            print(f"❌ PDF文件不存在: {pdf_path}")
            return
        
        # 测试2: 相似度搜索（仅记录日志，不打印到控制台）
        logger.info("2. 测试相似度搜索...")
        test_queries = [
            "客户经理考核标准是什么？",
            "客户经理每年评聘申报时间是怎样的？",
            "客户经理被投诉了，投诉一次扣多少分？"
        ]
        
        for query in test_queries:
            logger.info(f"查询: {query}")
            try:
                results = rag_system.search(query, k=3)
                logger.info(f"找到 {len(results)} 个搜索结果")
                for i, (content, score, page) in enumerate(results, 1):
                    logger.info(f"  {i}. 页码: {page}, 相似度: {score:.4f}")
                    logger.info(f"     内容: {content[:80]}...")
            except Exception as e:
                error_msg = handle_api_error(e)
                logger.error(f"❌ 搜索失败: {error_msg}")
        
        # 测试3: 智能问答（只显示问题和答案）
        print("\n" + "=" * 60)
        print("智能问答测试")
        print("=" * 60)
        
        questions = [
            "客户经理每年评聘申报时间是怎样的？",
            "客户经理考核标准是什么？",
            "客户经理被投诉了，投诉一次扣多少分？"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n问题 {i}: {question}")
            try:
                answer_info = rag_system.ask_question(question, k=5)
                print(f"答案: {answer_info['answer']}")
                print(f"来源页码: {answer_info['sources']}")
                print(f"API成本: {answer_info['cost']}")
                print(f"使用文档数: {answer_info['docs_used']}")
            except Exception as e:
                error_msg = handle_api_error(e)
                logger.error(f"❌ 问答失败: {error_msg}")
                print(f"❌ 问答失败: {error_msg}")
        
        # 测试4: 知识库保存和加载（仅记录日志）
        logger.info("4. 测试知识库持久化...")
        try:
            rag_system.vector_store_manager.save_knowledge_base(
                rag_system.knowledge_base, 
                "./test_knowledge_base"
            )
            logger.info("✅ 知识库保存成功")
            
            new_rag_system = RAGSystem(config)
            loaded_kb = new_rag_system.load_knowledge_base("./test_knowledge_base")
            logger.info("✅ 知识库加载成功")
            
            test_results = new_rag_system.search("考核标准", k=2)
            logger.info(f"加载的知识库搜索测试: 找到 {len(test_results)} 个结果")
            
        except Exception as e:
            logger.error(f"❌ 知识库持久化测试失败: {e}")
        
        # 测试完成总结
        print("\n" + "=" * 60)
        print("RAG系统测试完成！")
        print("=" * 60)
        logger.info("RAG系统测试完成！")
        
    except Exception as e:
        error_msg = handle_api_error(e)
        logger.error(f"❌ 测试过程中出现错误: {error_msg}")
        print(f"❌ 测试过程中出现错误: {error_msg}")


def main():
    """
    主函数示例
    演示如何使用RAG系统处理PDF文档并进行搜索
    
    功能:
    - 运行完整的系统测试
    - 验证所有功能模块
    - 提供测试报告
    """
    # 运行测试 - 执行完整的系统测试流程
    test_rag_system()


if __name__ == "__main__":
    # 程序入口点 - 当直接运行此文件时执行
    main()