import sys
import os

# 路径修正
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
# ✅ 同样引入免费的本地 Embedding
from langchain_huggingface import HuggingFaceEmbeddings 

# ✅ 引入配置 (确保 src/config.py 里有这些变量)
# 如果你 src/config.py 里没有 OPENAI_API_BASE，请记得去加上
from src.config import CHROMA_PATH, OPENAI_API_KEY, OPENAI_API_BASE

def main():
    print("🚀 正在启动医疗 Agent...")

    # 1. 准备 Embedding (必须和 build_db.py 用同一个模型)
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(CHROMA_PATH):
        print("❌ 错误: 数据库不存在，请先运行 python scripts/build_db.py")
        return
        
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 2. 准备大脑 (LLM)
    # 这里配置了 base_url，所以既支持 OpenAI，也支持 DeepSeek
    print(f"Connecting to LLM (Base URL: {OPENAI_API_BASE})...")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", # 如果用 DeepSeek，可以改成 "deepseek-chat"
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE 
    )
    
    # 3. 准备问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # 4. 开始交互
    print("\n✅ 系统就绪！我是你的全科医疗助手。")
    print("(输入 'quit' 或 'exit' 退出)")
    
    while True:
        try:
            query = input("\n👨‍⚕️ 请描述症状: ")
            if query.lower() in ['quit', 'exit']:
                print("👋 再见！")
                break
            
            if not query.strip():
                continue
                
            print("🤔 思考中...", end="", flush=True)
            result = qa_chain.invoke({"query": query})
            print("\r" + " " * 20 + "\r", end="") # 清除"思考中"
            
            print(f"🤖 AI 建议: \n{result['result']}")
            
            # 打印参考来源 (可选)
            # print("\n📚 参考文档:")
            # for doc in result['source_documents']:
            #     source = os.path.basename(doc.metadata.get('source', 'unknown'))
            #     print(f"- {source} (Page {doc.metadata.get('page', 0)})")

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main()