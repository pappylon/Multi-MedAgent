# RAG + Prompt Engine

from src.rag.loader import VectorDBLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

MEDICAL_PROMPT_TEMPLATE = """
You are a cautious and professional Medical AI Assistant.

Use the provided context to answer the question.

RULES:
1. If the question is clear and the answer can be found in the context:
   - Answer based on the context.
2. If the question is unclear or missing necessary information:
   - Politely ask a clarifying question to gather more details from the user.
3. If the answer is not in the context:
   - First, clearly state that the information was not found in the medical guidelines.
   - Then, provide general medical guidance or examples in a professional tone.
   - Always include a disclaimer: "This advice is for general reference only and does not constitute a medical diagnosis or professional consultation."
4. Maintain a clinical, professional, and empathetic tone.


Context:
{context}

Question:
{question}

Medical Answer:
"""

class GeminiRAGEngine:
    """RAG Engine using Google Gemini for medical QA"""

    def __init__(self, google_api_key: str, k: int = 5, temperature: float = 0):
        # 1️⃣ 加载向量数据库
        loader = VectorDBLoader(k=k)
        self.retriever = loader.load_db()

        # 2️⃣ 初始化 Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=google_api_key,
            temperature=temperature,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # 3️⃣ Prompt 模板
        self.prompt = PromptTemplate(
            template=MEDICAL_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def answer_question(self, question: str) -> str:
        # 检索相关文档
        docs = self.retriever.invoke(question)
        if not docs:
            return "⚠️ 知识库中未找到相关信息。"

        # 拼接上下文
        context_text = "\n\n---\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

        # 构造 Prompt
        full_prompt = self.prompt.format(context=context_text, question=question)

        # 调用 Gemini
        resp = self.llm.invoke(full_prompt)
        return getattr(resp, "content", str(resp))
