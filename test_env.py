import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 打印环境变量
print('DASHSCOPE_API_KEY:', os.getenv('DASHSCOPE_API_KEY'))
print('DINGTALK_WEBHOOK_URL:', os.getenv('DINGTALK_WEBHOOK_URL'))
print('WECHAT_WEBHOOK_URL:', os.getenv('WECHAT_WEBHOOK_URL'))