import streamlit as st
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# --------------------------------
# 路径设置：scripts → 项目根目录 → src
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from rag.engine import GeminiRAGEngine  # type: ignore

# --------------------------------
# 加载环境变量
# --------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------
# 页面标题
# --------------------------------
st.set_page_config(page_title="Medical AI Agent", page_icon="🏥")
st.title("🏥 Medical AI Assistant")
st.caption("Powered by Gemini 2.5 & RAG Technology")

# --------------------------------
# 初始化引擎
# --------------------------------
@st.cache_resource
def get_engine():
    if not GOOGLE_API_KEY:
        st.error("❌ 未找到 GOOGLE_API_KEY，请检查 .env 文件")
        return None
    return GeminiRAGEngine(google_api_key=GOOGLE_API_KEY)


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
# 输入栏
# --------------------------------
if prompt := st.chat_input("请描述您的症状或问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    if engine:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ 思考中...")

            try:
                # 转换为引擎所需格式
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
                placeholder.error(f"❌ 错误：{e}")
