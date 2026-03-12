import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量
load_dotenv()

print("测试模型初始化...")

# 测试语言模型
try:
    llm = ChatTongyi(
        model="qwen-plus",
        temperature=0.1,
        streaming=True,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    print("✓ 语言模型初始化成功")
    
    # 测试基本调用
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", "你好，请简单介绍一下自己")
    ])
    chain = prompt | llm | StrOutputParser()
    
    print("测试模型调用...")
    response = chain.invoke({})
    print(f"✓ 模型调用成功: {response[:50]}...")
except Exception as e:
    print(f"✗ 语言模型测试失败: {str(e)}")

# 测试嵌入模型
try:
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    print("✓ 嵌入模型初始化成功")
    
    # 测试基本调用
    embedding = embeddings.embed_query("测试嵌入")
    print(f"✓ 嵌入模型调用成功，嵌入维度: {len(embedding)}")
except Exception as e:
    print(f"✗ 嵌入模型测试失败: {str(e)}")

print("测试完成！")