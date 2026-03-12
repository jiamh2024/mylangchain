import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 加载环境变量
load_dotenv()

print("=== 全面功能测试 ===")

# 1. 测试环境变量
print("\n1. 测试环境变量:")
api_key = os.getenv("DASHSCOPE_API_KEY")
dingtalk_url = os.getenv("DINGTALK_WEBHOOK_URL")

if api_key:
    print("✓ DASHSCOPE_API_KEY 已设置")
else:
    print("✗ DASHSCOPE_API_KEY 未设置")

if dingtalk_url:
    print("✓ DINGTALK_WEBHOOK_URL 已设置")
else:
    print("✗ DINGTALK_WEBHOOK_URL 未设置")

# 2. 测试模型初始化
print("\n2. 测试模型初始化:")
try:
    llm = ChatTongyi(
        model="qwen-plus",
        temperature=0.1,
        streaming=True,
        dashscope_api_key=api_key
    )
    print("✓ 语言模型初始化成功")
except Exception as e:
    print(f"✗ 语言模型初始化失败: {str(e)}")

try:
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=api_key
    )
    print("✓ 嵌入模型初始化成功")
except Exception as e:
    print(f"✗ 嵌入模型初始化失败: {str(e)}")

# 3. 测试文档加载
print("\n3. 测试文档加载:")
try:
    # 创建测试文档
    test_dir = "data"
    os.makedirs(test_dir, exist_ok=True)
    
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档。\n测试内容：LangChain是一个强大的AI框架。")
    
    # 加载文档
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(test_file, encoding="utf-8")
    documents = loader.load()
    print(f"✓ 文档加载成功，共 {len(documents)} 个文档")
    print(f"文档内容: {documents[0].page_content[:50]}...")
except Exception as e:
    print(f"✗ 文档加载失败: {str(e)}")

# 4. 测试向量库
print("\n4. 测试向量库:")
try:
    # 分割文档
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    print(f"✓ 文档分割成功，共 {len(split_docs)} 个片段")
    
    # 创建向量库
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    print("✓ 向量库创建成功")
    
    # 测试检索
    query = "LangChain"
    docs = vectorstore.similarity_search(query, k=1)
    print(f"✓ 向量库检索成功，找到 {len(docs)} 个相关文档")
    print(f"检索结果: {docs[0].page_content[:50]}...")
except Exception as e:
    print(f"✗ 向量库测试失败: {str(e)}")

# 5. 测试钉钉消息发送
print("\n5. 测试钉钉消息发送:")
try:
    import requests
    
    if not dingtalk_url:
        print("⚠ 钉钉Webhook URL未设置，跳过测试")
    else:
        data = {
            "msgtype": "text",
            "text": {
                "content": "(xiaojia) 测试消息：知识库助手功能正常！"
            }
        }
        response = requests.post(dingtalk_url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("errcode") == 0:
                print("✓ 钉钉消息发送成功")
            else:
                print(f"✗ 钉钉消息发送失败: {result.get('errmsg', '未知错误')}")
        else:
            print(f"✗ 钉钉消息发送失败: HTTP {response.status_code}")
except Exception as e:
    print(f"✗ 钉钉消息发送测试失败: {str(e)}")

print("\n=== 测试完成 ===")