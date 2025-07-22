import logging
import io
import os
from qwen_agent.agents import Assistant

# =========================
# 日志捕获工具类
# =========================
class LogCapture:
    """
    自定义日志捕获器，用于捕获qwen_agent和根日志的输出，便于后续分析。
    """
    def __init__(self):
        self.log_capture_string = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture_string)
        self.log_handler.setLevel(logging.INFO)
        self.log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(self.log_formatter)
        
        # 捕获qwen_agent日志
        self.logger = logging.getLogger('qwen_agent_logger')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_handler)
        
        # 捕获根日志
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.INFO)
        self.root_logger.addHandler(self.log_handler)
    
    def get_log(self):
        """获取当前捕获的日志内容"""
        return self.log_capture_string.getvalue()
    
    def clear_log(self):
        """清空日志内容"""
        self.log_capture_string.truncate(0)
        self.log_capture_string.seek(0)

# =========================
# 初始化日志捕获器
# =========================
log_capture = LogCapture()

# =========================
# 步骤 1：配置 LLM
# =========================
llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    'model': 'qwen-max',
    'model_server': 'dashscope',
    'api_key': os.getenv("ALI_API_KEY"),
    'generate_cfg': {
        'top_p': 0.8
    }
}

# =========================
# 步骤 2：创建智能体
# =========================
system_instruction = ''  # 系统提示词，可自定义
# 工具列表，可扩展
tools = []
# 需要让智能体读取的文件列表
files = ['./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf']

# 清除之前的日志，保证本次运行日志干净
log_capture.clear_log()

# 创建 Assistant 智能体
bot = Assistant(
    llm=llm_cfg,
    system_message=system_instruction,
    function_list=tools,
    files=files
)

# =========================
# 步骤 3：运行智能体进行问答
# =========================
messages = []  # 聊天历史
query = "客户经理被客户投诉一次，扣多少分？"
messages.append({'role': 'user', 'content': query})
response = []
current_index = 0

# 运行智能体，逐步输出响应
for response in bot.run(messages=messages):
    if current_index == 0:
        # 第一次响应时，分析日志，提取检索相关内容
        log_content = log_capture.get_log()
        print("\n===== 从日志中提取的检索信息 =====")
        # 检索相关日志
        retrieval_logs = [line for line in log_content.split('\n') 
                         if any(keyword in line.lower() for keyword in 
                               ['retriev', 'search', 'chunk', 'document', 'ref', 'token'])]
        for log_line in retrieval_logs:
            print(log_line)
        # 可能包含文档内容的日志
        content_logs = [line for line in log_content.split('\n') 
                       if any(keyword in line.lower() for keyword in 
                             ['content', 'text', 'document', 'chunk'])]
        print("\n===== 可能包含文档内容的日志 =====")
        for log_line in content_logs:
            print(log_line)
        print("===========================\n")
    # 增量输出当前响应内容
    current_response = response[0]['content'][current_index:]
    current_index = len(response[0]['content'])
    print(current_response, end='')

# 将机器人的回应添加到聊天历史
messages.extend(response)

# =========================
# 运行结束后日志分析
# =========================
print("\n\n===== 运行结束后的完整日志分析 =====")
log_content = log_capture.get_log()

# 1. 关键词提取相关日志
print("\n1. 关键词提取:")
keyword_logs = [line for line in log_content.split('\n') if 'keywords' in line.lower()]
for log_line in keyword_logs:
    print(log_line)

# 2. 文档处理相关日志
print("\n2. 文档处理:")
doc_logs = [line for line in log_content.split('\n') if 'doc' in line.lower() or 'chunk' in line.lower()]
for log_line in doc_logs:
    print(log_line)

# 3. 检索相关日志
print("\n3. 检索相关:")
retrieval_logs = [line for line in log_content.split('\n') if 'retriev' in line.lower() or 'search' in line.lower() or 'ref' in line.lower()]
for log_line in retrieval_logs:
    print(log_line)

# 4. 可能包含文档内容的日志
print("\n4. 可能包含文档内容的日志:")
content_logs = [line for line in log_content.split('\n') if 'content:' in line.lower() or 'text:' in line.lower()]
for log_line in content_logs:
    print(log_line)

print("===========================\n")
