import sys
import os
from langchain_core.prompts import PromptTemplate
from src.rag.loader import VectorDBLoader
from src.rag.config import MEDICAL_PROMPT_TEMPLATE

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
fine_tune_dir = os.path.join(project_root, "fine-tune")
if fine_tune_dir not in sys.path:
    sys.path.append(fine_tune_dir)
    print(f"✅ 已添加模型路径: {fine_tune_dir}")

try:
    from inference import load_local_model, generate_local_response
except ImportError:
    print("⚠️ 无法导入 inference.py")

class LocalRAGEngine:
    def __init__(self, k: int = 3):
        # 1. RAG 检索部分
        print("🔍 初始化检索器...")
        loader = VectorDBLoader(k=k)
        self.retriever = loader.load_db()

        # 2. 加载模型
        # model, tokenizer, device 三个变量都要接住
        self.model, self.tokenizer, self.device = load_local_model()

        # 3. Prompt 模板
        self.prompt = PromptTemplate(
            template=MEDICAL_PROMPT_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )

    def answer_question(self, question: str, chat_history: list = None) -> str:
        if not self.model:
            return "❌ 模型加载失败，无法回答。"

        # 1. 检索
        print(f"🔍 [RAG] 正在检索: {question}")
        docs = self.retriever.invoke(question)
        context_text = "\n".join([d.page_content for d in docs])

        # 2. 历史记录 (取最近2轮)
        history_text = ""
        if chat_history:
            for role, text in chat_history[-2:]:
                history_text += f"{role}: {text}\n"

        # 3. 组装符合训练格式的 Prompt
        full_prompt = self.prompt.format(
            context=context_text,
            chat_history=history_text,
            question=question
        )

        # 4. 生成
        print("🧠 [Local Model] 正在生成回答...")
        response = generate_local_response(self.model, self.tokenizer, self.device, full_prompt)
        
        return response