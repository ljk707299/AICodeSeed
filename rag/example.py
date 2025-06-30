#!/usr/bin/env python3
"""
RAG系统使用示例
演示如何使用RAG系统处理PDF文档并进行问答
"""

from main import RAGSystem, RAGConfig, check_api_keys, handle_api_error

def simple_example():
    """简单使用示例"""
    print("=" * 50)
    print("RAG系统简单使用示例")
    print("=" * 50)
    
    # 检查API密钥
    api_status = check_api_keys()
    if not api_status['any_configured']:
        print("❌ 请先配置API密钥")
        return
    
    # 创建配置
    config = RAGConfig(
        chunk_size=512,
        chunk_overlap=128,
        save_path="./example_knowledge_base"
    )
    
    # 创建RAG系统
    rag_system = RAGSystem(config)
    
    # PDF文件路径
    pdf_path = './浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf'
    
    try:
        # 处理PDF文件
        print(f"\n📄 处理PDF文件: {pdf_path}")
        knowledge_base = rag_system.process_pdf(pdf_path)
        print(f"✅ 知识库创建成功，包含 {len(knowledge_base.page_info)} 个文本块")
        
        # 示例问题
        questions = [
            "客户经理考核标准是什么？",
            "客户经理每年评聘申报时间是怎样的？",
            "客户经理被投诉了，投诉一次扣多少分？"
        ]
        
        # 进行问答
        for question in questions:
            print(f"\n❓ 问题: {question}")
            try:
                answer_info = rag_system.ask_question(question, k=3)
                print(f"💡 答案: {answer_info['answer']}")
                print(f"📖 来源页码: {answer_info['sources']}")
                print(f"💰 API成本: {answer_info['cost']}")
            except Exception as e:
                error_msg = handle_api_error(e)
                print(f"❌ 问答失败: {error_msg}")
        
        print("\n🎉 示例运行完成！")
        
    except Exception as e:
        error_msg = handle_api_error(e)
        print(f"❌ 运行失败: {error_msg}")

def search_example():
    """搜索示例"""
    print("\n" + "=" * 50)
    print("搜索功能示例")
    print("=" * 50)
    
    # 创建配置
    config = RAGConfig()
    rag_system = RAGSystem(config)
    
    try:
        # 加载已存在的知识库
        print("📚 加载知识库...")
        rag_system.load_knowledge_base("./example_knowledge_base")
        
        # 搜索示例
        query = "考核标准"
        print(f"\n🔍 搜索: {query}")
        
        results = rag_system.search(query, k=3)
        print(f"找到 {len(results)} 个相关结果:")
        
        for i, (content, score, page) in enumerate(results, 1):
            print(f"\n{i}. 页码: {page}, 相似度: {score:.4f}")
            print(f"   内容: {content[:100]}...")
        
    except Exception as e:
        error_msg = handle_api_error(e)
        print(f"❌ 搜索失败: {error_msg}")

if __name__ == "__main__":
    # 运行简单示例
    simple_example()
    
    # 运行搜索示例
    search_example() 