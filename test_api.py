import os
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    print("错误：未设置DASHSCOPE_API_KEY")
    exit(1)

print(f"API密钥: {api_key[:8]}...")

# 测试DashScope API连接
try:
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "qwen-plus",
        "input": {
            "prompt": "你好"
        },
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.7
        }
    }
    
    print("测试API连接...")
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    
    if response.status_code == 200:
        print("✓ API连接成功")
    else:
        print("✗ API连接失败")
        
except Exception as e:
    print(f"✗ API测试失败: {str(e)}")

print("测试完成！")