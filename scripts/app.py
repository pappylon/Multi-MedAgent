import streamlit as st
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# --------------------------------
# 路径设置
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


# from rag.engine import GeminiRAGEngine
from rag.engine import LocalLLMEngine


# --------------------------------
# 加载环境变量
# --------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------
# 页面配置
# --------------------------------
st.set_page_config(page_title="Medical AI Agent (Local)", page_icon="🏥")
st.title("🏥 Medical AI Assistant")
# ✅ 改动 2: 更新副标题，强调使用了微调模型
st.caption("🚀 Powered by **Local Fine-tuned Model** (Llama-3) & RAG Technology")

# --------------------------------
# 初始化引擎 (带缓存)
# --------------------------------
@st.cache_resource
def get_engine():
    # 显示一个加载转圈圈，因为本地加载比较慢
    with st.spinner("正在加载本地微调模型 (约需 1-2 分钟，请耐心等待)..."):
        try:
            # ✅ 改动 3: 实例化本地引擎
            return LocalLLMEngine()
        except Exception as e:
            st.error(f"❌ 模型加载失败: {e}")
            return None

# 执行加载
engine = get_engine()

# --------------------------------
# Session State 聊天历史
# --------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------
# 处理用户输入
# --------------------------------
if prompt := st.chat_input("Please describe your symptoms or concerns..."):
    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})


    if engine:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ *Local Model is thinking...*")

            try:

                history_for_engine = [
                    ("User" if m["role"] == "user" else "AI", m["content"])
                    for m in st.session_state.messages[:-1]
                ]

                response = engine.answer_question(prompt, history_for_engine)

                placeholder.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                placeholder.error(f"❌Runtime error:{e}")
    else:
        st.error("⚠️ The engine has not been initialized and cannot respond. Please check the model path.")