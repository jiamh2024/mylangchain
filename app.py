from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import os
import requests
import threading
from flask import Flask, request, jsonify
import time

# 加载环境变量
load_dotenv()

# 创建Flask应用
app = Flask(__name__)

# 初始化语言模型（使用openRouter平台的大模型）
llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    temperature=0.1,
    streaming=True,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 初始化嵌入模型（使用OpenRouter平台的nvidia/llama-nemotron-embed-vl-1b-v2:free模型）
embeddings = OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 发送消息到钉钉的工具
@tool
def send_dingtalk_message(message: str) -> str:
    """
    发送消息到钉钉。当用户要求将回答发送到钉钉时使用此工具。
    
    参数:
        message: 要发送的消息内容
    
    返回:
        发送结果
    """
    webhook_url = os.getenv("DINGTALK_WEBHOOK_URL")
    
    if not webhook_url:
        return "错误：未配置钉钉Webhook URL，请在.env文件中设置DINGTALK_WEBHOOK_URL"
    
    try:
        data = {
            "msgtype": "text",
            "text": {
                "content": f"(xiaojia) 知识库助手回答：\n{message}"
            }
        }
        response = requests.post(webhook_url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("errcode") == 0:
                return "消息已成功发送到钉钉"
            else:
                return f"发送到钉钉失败：{result.get('errmsg', '未知错误')}"
        else:
            return f"发送到钉钉失败：HTTP {response.status_code}"
    except Exception as e:
        return f"发送到钉钉失败：{str(e)}"

# 发送消息到微信的工具
@tool
def send_wechat_message(message: str) -> str:
    """
    发送消息到微信。当用户要求将回答发送到微信时使用此工具。
    
    参数:
        message: 要发送的消息内容
    
    返回:
        发送结果
    """
    webhook_url = os.getenv("WECHAT_WEBHOOK_URL")
    
    if not webhook_url:
        return "错误：未配置微信Webhook URL，请在.env文件中设置WECHAT_WEBHOOK_URL"
    
    try:
        data = {
            "msgtype": "text",
            "text": {
                "content": f"知识库助手回答：\n{message}"
            }
        }
        response = requests.post(webhook_url, json=data, timeout=10)
        if response.status_code == 200:
            return "消息已成功发送到微信"
        else:
            return f"发送到微信失败：HTTP {response.status_code}"
    except Exception as e:
        return f"发送到微信失败：{str(e)}"

# 根路径路由
@app.route('/')
def index():
    """
    根路径路由，返回系统状态信息
    """
    return jsonify({
        'status': 'running',
        'service': '知识库助手',
        'version': '4.0.0',
        'integrations': {
            'dingtalk': True,
            'wechat': True
        }
    })

# 可用工具列表
tools = [send_dingtalk_message, send_wechat_message]

# 加载文档
def load_documents():
    documents = []
    data_dir = "data"
    
    # 使用os.walk递归遍历所有子目录
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            try:
                if filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents.extend(loader.load())
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif filename.endswith(".html") or filename.endswith(".htm"):
                    loader = BSHTMLLoader(file_path, open_encoding="utf-8")
                    documents.extend(loader.load())
            except Exception as e:
                print(f"加载文件失败: {file_path}, 错误: {str(e)}")
                continue
    
    return documents

# 分割文档
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 向量库存储路径
VECTORSTORE_PATH = "faiss_index"

# 创建向量存储
def create_vectorstore(force_rebuild=False):
    # 检查是否需要重建向量库
    if not force_rebuild and os.path.exists(VECTORSTORE_PATH):
        print("加载已保存的向量库...")
        try:
            return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"加载向量库失败，将重新创建: {str(e)}")
            force_rebuild = True
    
    print("创建新的向量库...")
    documents = load_documents()
    split_docs = split_documents(documents)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    # 保存向量库
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"向量库已保存到: {VECTORSTORE_PATH}")
    
    return vectorstore

# 初始化向量存储
vectorstore = create_vectorstore()

# 创建Agent提示模板
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识库助手，能够回答关于上传文档的问题。
    
你有两个可用的工具：
1. send_dingtalk_message: 发送消息到钉钉
2. send_wechat_message: 发送消息到微信

当用户要求将回答发送到钉钉或微信时，必须使用相应的工具。
不要自己发送消息，必须调用工具来发送。

如果用户没有要求发送到特定平台，则直接回答问题即可。
"""),
    ("human", "{question}")
])

# 创建Agent
tools_with_context = [send_dingtalk_message, send_wechat_message]
agent = create_agent(
    model=llm,
    tools=tools_with_context,
    system_prompt="你是一个知识库助手，能够回答关于上传文档的问题。你有两个可用的工具：1. send_dingtalk_message: 发送消息到钉钉 2. send_wechat_message: 发送消息到微信。当用户要求将回答发送到特定平台时，必须使用相应的工具。不要自己发送消息，必须调用工具来发送。如果用户没有要求发送到特定平台，则直接回答问题即可。"
)

# 创建输出解析器
output_parser = StrOutputParser()

# 创建链
chain = agent_prompt | llm | output_parser

# 主函数
def main():
    global vectorstore
    
    print("欢迎使用知识库聊天机器人！")
    print("输入 'exit' 退出程序。")
    print("输入 'rebuild' 重新构建向量库。")
    print("输入问题后，如果需要发送到钉钉或微信，请说明。")
    
    while True:
        question = input("请输入您的问题：")
        
        if question.lower() == "exit":
            print("再见！")
            break
        
        if question.lower() == "rebuild":
            print("重新构建向量库...")
            vectorstore = create_vectorstore(force_rebuild=True)
            print("向量库已重新构建完成！")
            continue
        
        try:
            # 检索相关文档
            docs = vectorstore.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # 检查是否需要发送到特定平台
            question_lower = question.lower()
            need_send = "钉钉" in question_lower or "微信" in question_lower or "dingtalk" in question_lower or "wechat" in question_lower
            
            # 构建完整的问题，包含上下文
            full_question = f"根据以下文档内容回答问题：\n\n文档内容：\n{context}\n\n用户问题：{question}"
            
            if need_send:
                # 使用Agent处理，包含发送消息功能
                inputs = {"messages": [{"role": "user", "content": full_question}]}
                response = ""
                for chunk in agent.stream(inputs, stream_mode="updates"):
                    if "messages" in chunk:
                        for msg in chunk["messages"]:
                            if msg.role == "assistant":
                                response = msg.content
                print(f"回答：{response}")
            else:
                # 使用普通链处理
                prompt_with_context = ChatPromptTemplate.from_messages([
                    ("system", "你是一个知识库助手，能够回答关于上传文档的问题。请根据以下文档内容进行回答，不要编造信息。\n\n{context}"),
                    ("human", "{question}")
                ])
                chain_with_context = prompt_with_context | llm | output_parser
                response = chain_with_context.invoke({"question": question, "context": context})
                print(f"回答：{response}")
        except Exception as e:
            print(f"错误：{str(e)}")

# 运行Flask服务器
def run_flask():
    """
    运行Flask服务器，用于接收Webhook
    """
    port = int(os.getenv("FLASK_PORT", 5000))
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    print(f"Flask服务器启动在 http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    # 启动Flask服务器线程
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # 运行主函数
    main()
