# src/rag/engine.py
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.prompts import PromptTemplate
from rag.loader import VectorDBLoader
from rag.config import MEDICAL_PROMPT_TEMPLATE  # ✅ 从配置文件导入

class GeminiRAGEngine:
    """RAG Engine using Google Gemini for medical QA with Memory"""

    def __init__(self, google_api_key: str, k: int = 5, temperature: float = 0):
        # 1️⃣ 加载向量数据库
        loader = VectorDBLoader(k=k)
        self.retriever = loader.load_db()

        # 2️⃣ 初始化 Gemini LLM
        # 建议：使用 standard 'gemini-2.5-flash' 进行更好的推理，如果不缺配额的话
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=google_api_key,
            temperature=temperature,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # 3️⃣ Prompt 模板 (增加了 chat_history)
        self.prompt = PromptTemplate(
            template=MEDICAL_PROMPT_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )

    def answer_question(self, question: str, chat_history: list = None) -> str:
        # 检索相关文档
        docs = self.retriever.invoke(question)
        if not docs:
            return "⚠️ 知识库中未找到相关信息。"

        # 拼接上下文
        context_text = "\n\n---\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

        # ✅ 处理历史记录：将列表转换为字符串
        history_text = ""
        if chat_history:
            # 只取最近 3 轮对话，避免 Token 溢出
            for role, text in chat_history[-6:]:
                history_text += f"{role}: {text}\n"
        else:
            history_text = "No previous conversation."

        # 构造 Prompt
        full_prompt = self.prompt.format(
            context=context_text, 
            chat_history=history_text, 
            question=question
        )

        # 调用 Gemini
        resp = self.llm.invoke(full_prompt)
        return getattr(resp, "content", str(resp))