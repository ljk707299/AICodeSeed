#!/usr/bin/env python3
"""
RAG系统快速设置脚本
帮助用户快速配置环境和API密钥
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    print(f"✅ Python版本检查通过: {sys.version}")
    return True

def install_dependencies():
    """安装依赖包"""
    print("\n📦 安装依赖包...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def setup_api_keys():
    """设置API密钥"""
    print("\n🔑 配置API密钥...")
    
    # 检查是否已有环境变量
    dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if dashscope_key:
        print("✅ 阿里百炼API密钥已配置")
    if openai_key:
        print("✅ OpenAI API密钥已配置")
    
    if not (dashscope_key or openai_key):
        print("⚠️  未检测到API密钥配置")
        print("\n请选择配置方式:")
        print("1. 手动设置环境变量")
        print("2. 创建.env文件")
        
        choice = input("\n请选择 (1/2): ").strip()
        
        if choice == "2":
            create_env_file()
        else:
            print("\n请手动设置环境变量:")
            print("export DASHSCOPE_API_KEY='your_api_key'")
            print("或")
            print("export OPENAI_API_KEY='your_api_key'")
    
    return True

def create_env_file():
    """创建.env文件"""
    print("\n📝 创建.env文件...")
    
    env_content = """# RAG系统环境变量配置

# 阿里百炼API密钥 (推荐)
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# 或者使用ALI_API_KEY
# ALI_API_KEY=your_ali_api_key_here

# OpenAI API密钥 (备选)
# OPENAI_API_KEY=your_openai_api_key_here

# 日志级别
LOG_LEVEL=INFO
"""
    
    try:
        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_content)
        print("✅ .env文件创建成功")
        print("⚠️  请编辑.env文件，填入您的实际API密钥")
        return True
    except Exception as e:
        print(f"❌ 创建.env文件失败: {e}")
        return False

def check_pdf_file():
    """检查PDF文件"""
    print("\n📄 检查PDF文件...")
    pdf_path = Path("./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf")
    
    if pdf_path.exists():
        print(f"✅ 找到PDF文件: {pdf_path}")
        return True
    else:
        print(f"⚠️  未找到PDF文件: {pdf_path}")
        print("请将PDF文件放在当前目录下")
        return False

def run_test():
    """运行测试"""
    print("\n🧪 运行系统测试...")
    try:
        from main import test_rag_system
        test_rag_system()
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("RAG系统快速设置")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 安装依赖
    if not install_dependencies():
        return
    
    # 设置API密钥
    if not setup_api_keys():
        return
    
    # 检查PDF文件
    check_pdf_file()
    
    # 询问是否运行测试
    print("\n" + "=" * 60)
    print("设置完成！")
    print("=" * 60)
    
    run_test_choice = input("\n是否运行系统测试？(y/n): ").strip().lower()
    if run_test_choice in ['y', 'yes', '是']:
        run_test()
    
    print("\n🎉 设置完成！")
    print("现在可以运行: python main.py")

if __name__ == "__main__":
    main() 