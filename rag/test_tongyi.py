#!/usr/bin/env python3
"""
测试通义千问集成
验证RAG系统与通义千问的兼容性
"""

import warnings
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from main import RAGSystem, RAGConfig, check_api_keys, handle_api_error

def test_tongyi_integration():
    """测试通义千问集成"""
    print("=" * 60)
    print("测试通义千问集成")
    print("=" * 60)
    
    # 检查API密钥
    api_status = check_api_keys()
    if not api_status['any_configured']:
        print("❌ 请先配置API密钥")
        return
    
    # 创建配置（使用通义千问）
    config = RAGConfig(
        chunk_size=512,
        chunk_overlap=128,
        llm_model="qwen-turbo",  # 使用通义千问
        save_path="./tongyi_test_kb"
    )
    
    # 创建RAG系统
    rag_system = RAGSystem(config)
    
    # PDF文件路径
    pdf_path = './浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf'
    
    try:
        # 处理PDF文件
        print(f"\n📄 处理PDF文件: {pdf_path}")
        if not Path(pdf_path).exists():
            print(f"❌ PDF文件不存在: {pdf_path}")
            return
            
        knowledge_base = rag_system.process_pdf(pdf_path)
        print(f"✅ 知识库创建成功，包含 {len(knowledge_base.page_info)} 个文本块")
        
        # 测试问答功能
        print("\n🧪 测试通义千问问答功能...")
        questions = [
            "客户经理考核标准是什么？",
            "客户经理每年评聘申报时间是怎样的？",
            "客户经理被投诉了，投诉一次扣多少分？"
        ]
        
        for question in questions:
            print(f"\n❓ 问题: {question}")
            
            # 捕获警告
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    answer_info = rag_system.ask_question(question, k=3)
                    
                    # 检查是否有弃用警告
                    deprecation_warnings = [warning for warning in w if "deprecated" in str(warning.message).lower()]
                    
                    if deprecation_warnings:
                        print("⚠️  存在弃用警告:")
                        for warning in deprecation_warnings:
                            print(f"  - {warning.message}")
                    else:
                        print("✅ 没有弃用警告！")
                    
                    print(f"💡 答案: {answer_info['answer']}")
                    print(f"📖 来源页码: {answer_info['sources']}")
                    print(f"💰 API成本: {answer_info['cost']}")
                    
                except Exception as e:
                    error_msg = handle_api_error(e)
                    print(f"❌ 问答失败: {error_msg}")
        
        print("\n🎉 通义千问集成测试完成！")
        
    except Exception as e:
        error_msg = handle_api_error(e)
        print(f"❌ 测试失败: {error_msg}")

def test_llm_initialization():
    """测试LLM初始化"""
    print("\n🔧 测试LLM初始化...")
    
    try:
        from main import LLMManager, RAGConfig
        
        config = RAGConfig(llm_model="qwen-turbo")
        llm_manager = LLMManager(config)
        
        llm = llm_manager.get_llm()
        print(f"✅ LLM初始化成功: {type(llm).__name__}")
        
        # 测试简单调用
        test_prompt = "你好，请简单介绍一下自己。"
        response = llm.invoke(test_prompt)
        print(f"✅ LLM调用成功: {str(response)[:100]}...")
        
    except Exception as e:
        print(f"❌ LLM初始化失败: {e}")

if __name__ == "__main__":
    # 测试LLM初始化
    test_llm_initialization()
    
    # 测试通义千问集成
    test_tongyi_integration() 