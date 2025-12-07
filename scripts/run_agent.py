import sys
import os
import textwrap

# Ensure src module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.prompts import PromptTemplate
from src.config import CHROMA_PATH, OPENAI_API_KEY, OPENAI_API_BASE

# --- 1. Define a Medical Guardrail Prompt ---
# This strictly tells the AI to ONLY use the provided context.
MEDICAL_PROMPT_TEMPLATE = """
You are a helpful and cautious Medical AI Assistant. 
Use the following pieces of retrieved context to answer the user's question.

RULES:
1. If the answer is not in the context, strictly state: "I cannot find this information in my medical guidelines."
2. Do not make up answers or use outside knowledge.
3. Always maintain a professional, clinical tone.

Context:
{context}

Question:
{question}

Medical Answer:
"""

def main():
    print("🚀 启动医疗 Agent (Medical Mode)...")

    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Check Database
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ 数据库路径 {CHROMA_PATH} 不存在，请先运行 build_db.py")
        return
        
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 3. Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0, # Keep temp at 0 for medical facts (determinism)
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )

    # 4. Set up Retriever
    # increased k=5 to get more context for complex medical queries
    retriever = db.as_retriever(search_kwargs={"k": 5}) 

    # 5. Set up Chain with Custom Prompt
    PROMPT = PromptTemplate(
        template=MEDICAL_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )
    
    # We pass the prompt to the chain here
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    print("✅ 医疗助手就绪！（输入 'quit' 退出）")
    print("⚠️  免责声明：本工具仅供参考，不构成专业医疗建议。")

    while True:
        try:
            query = input("\n👨‍⚕️ 请输入医疗问题： ").strip()
            
            if not query:
                continue
                
            if query.lower() in ["quit", "exit"]:
                print("👋 Exiting...")
                break

            # 6. Retrieve Docs (Modern Syntax)
            # 'invoke' is preferred over 'get_relevant_documents' in newer versions
            docs = retriever.invoke(query)
            
            if not docs:
                print("⚠️ 数据库中未找到相关文档。")
                continue

            # 7. Generate Answer
            # Using .invoke instead of .run
            result = qa_chain.invoke({"input_documents": docs, "question": query})

            print("\n🤖 AI 回答：")
            # formatting text for easier reading
            print(textwrap.fill(result["output_text"], width=80))
            
            # Optional: Show source of information
            # print(f"\n📄 来源: {docs[0].metadata['source']}")

        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()