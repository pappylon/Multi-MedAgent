# scripts/app.py

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# 1. 路径设置：确保能找到 src 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.engine import GeminiRAGEngine

# 2. 加载环境变量
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 3. 页面配置
st.set_page_config(page_title="Medical AI Agent", page_icon="🏥", layout="centered")
st.title("🏥 Medical AI Assistant")
st.caption("Powered by Gemini 2.5 & RAG Technology")

# 4. 初始化引擎 (使用缓存装饰器，避免每次交互都重新加载向量库)
@st.cache_resource
def get_engine():
    if not GOOGLE_API_KEY:
        st.error("❌ 未找到 GOOGLE_API_KEY，请检查 .env 文件")
        return None
    return GeminiRAGEngine(google_api_key=GOOGLE_API_KEY)

engine = get_engine()

# 5. 初始化聊天历史 (Session State)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 6. 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. 处理用户输入
if prompt := st.chat_input("请描述您的症状或问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    # 保存用户消息到状态
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 准备调用引擎
    if engine:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ *Thinking...*")
            
            try:
                # --- 关键步骤：格式转换 ---
                # 将 Streamlit 的字典格式 [{"role": "user",...}] 
                # 转换为 Engine 需要的元组列表 [("User", "msg")...]
                history_for_engine = [
                    ("User" if m["role"] == "user" else "AI", m["content"])
                    for m in st.session_state.messages[:-1] # 不包含当前这句，防止重复
                ]

                # 调用你的 RAG 引擎
                response = engine.answer_question(prompt, history_for_engine)
                
                # 显示回答
                message_placeholder.markdown(response)
                
                # 保存 AI 回答到状态
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                message_placeholder.error(f"❌ Error: {str(e)}")