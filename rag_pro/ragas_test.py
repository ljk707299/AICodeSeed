"""
RAGas 评测脚本
----------------
本脚本用于评测RAG系统的问答效果，支持自动化批量问答、上下文检索、结果评测。
结构分为：环境初始化、文档处理、问答链构建、批量推理、评测。
"""

# ========== 1. 环境初始化 ========== #
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# 1.1 读取API密钥
DASHSCOPE_API_KEY = os.getenv("ALI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置ALI_API_KEY或DASHSCOPE_API_KEY环境变量")

# 1.2 初始化大语言模型和嵌入模型
llm = Tongyi(
    model_name="qwen-max",
    dashscope_api_key=DASHSCOPE_API_KEY
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# ========== 2. 文档处理与向量库 ========== #
# 2.1 文本分块器（父块/子块）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)

# 2.2 向量数据库与内存存储
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings
)
store = InMemoryStore()

# 2.3 父文档检索器（支持父子块检索）
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2}
)

# 2.4 添加文档（请替换为实际文档列表docs）
# docs = [...]  # 这里应为List[Document]，如从PDF/文本加载
# retriever.add_documents(docs)
# print(f"已添加文档数: {len(list(store.yield_keys()))}")

# ========== 3. 问答链构建 ========== #
# 3.1 Prompt模板
prompt_template = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use two sentences maximum and keep the answer concise.\nQuestion: {question}\nContext: {context}\nAnswer:\n"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 3.2 LCEL链式表达：检索→构建prompt→LLM→输出解析
chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

# ========== 4. 批量推理与上下文检索 ========== #
# 4.1 问题与标准答案
questions = [
    "客户经理被投诉了，投诉一次扣多少分？",
    "客户经理每年评聘申报时间是怎样的？",
    "客户经理在工作中有不廉洁自律情况的，发现一次扣多少分？",
    "客户经理不服从支行工作安排，每次扣多少分？",
    "客户经理需要什么学历和工作经验才能入职？",
    "个金客户经理职位设置有哪些？"
]
ground_truths = [
    "每投诉一次扣2分",
    "每年一月份为客户经理评聘的申报时间",
    "在工作中有不廉洁自律情况的每发现一次扣50分",
    "不服从支行工作安排，每次扣2分",
    "须具备大专以上学历，至少二年以上银行工作经验",
    "个金客户经理职位设置为：客户经理助理、客户经理、高级客户经理、资深客户经理"
]

# 4.2 批量问答与上下文检索
answers = []
contexts = []
for query in questions:
    # 生成答案
    answers.append(chain.invoke({"question": query}))
    # 检索上下文内容
    contexts.append([doc.page_content for doc in retriever.invoke(query)])

# ========== 5. 构建评测数据集与自动评测 ========== #
# 5.1 构建评测数据集
ragas_data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
}
dataset = Dataset.from_dict(ragas_data)

# 5.2 RAGas自动评测
result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    embeddings=embeddings
)

# 5.3 输出评测结果
print("\n===== RAGas自动评测结果 =====")
print(result)
print("\n===== 详细分数（DataFrame） =====")
print(result.to_pandas())

