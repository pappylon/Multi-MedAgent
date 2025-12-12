import sys
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

# 导入配置和加载器
from rag.loader import VectorDBLoader
from rag.config import MEDICAL_PROMPT_TEMPLATE, REWRITE_PROMPT_TEMPLATE

# ========================================================
# 1. 动态添加 fine-tune 路径
# ========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
fine_tune_dir = os.path.join(project_root, "fine-tune")

if fine_tune_dir not in sys.path:
    sys.path.append(fine_tune_dir)

try:
    from inference import load_local_model, generate_local_response
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False

# ========================================================
# 2. GeminiRAGEngine
# ========================================================
class GeminiRAGEngine:
    def __init__(self, google_api_key: str, k: int = 5, temperature: float = 0):
        # 1. 检索器
        print("🔍 初始化 Gemini RAG 检索器...")
        loader = VectorDBLoader(k=k)
        self.retriever = loader.load_db()

        # 2. LLM 初始化
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
        
        # 3. 回答模板
        self.prompt = PromptTemplate(
            template=MEDICAL_PROMPT_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )
        
        # 4. ✅ 关键修复：加载重写模板 (LangChain 链式调用)
        # 这会让 Gemini 先把 "it" 翻译成 "headache"
        rewrite_prompt_template = PromptTemplate(
            template=REWRITE_PROMPT_TEMPLATE,
            input_variables=["chat_history", "question"]
        )
        # 构造链：Prompt -> LLM -> String
        self.rewrite_chain = rewrite_prompt_template | self.llm | StrOutputParser()

    def rewrite_query(self, question: str, chat_history: list) -> str:
        """利用 Gemini 自身的高智商来重写问题"""
        if not chat_history:
            return question
            
        # 格式化历史记录
        history_text = "\n".join([f"{role}: {text}" for role, text in chat_history[-3:]])
        
        try:
            # 执行重写
            new_q = self.rewrite_chain.invoke({
                "chat_history": history_text,
                "question": question
            })
            # 打印出来，让你确认它是否工作
            # print(f"🔄 [Gemini Rewrite] '{question}' -> '{new_q.strip()}'")
            return new_q.strip()
        except Exception as e:
            print(f"⚠️ 重写失败: {e}")
            return question

    def answer_question(self, question: str, chat_history: list = None) -> str:
        # 1. 第一步：重写 (Rewriting)
        # "How to solve it?" -> "How to solve headache?"
        search_query = self.rewrite_query(question, chat_history)
        
        # 2. 第二步：检索 (Retrieval)
        # 用 "headache" 去搜，肯定能搜到
        print(f"🔍 [Gemini RAG] 正在检索: {search_query}")
        docs = self.retriever.invoke(search_query)

        # ==================== 🛠️ Debug 代码开始 ====================
        print(f"📄 [Debug] 检索到了 {len(docs)} 个相关片段")
        if len(docs) > 0:
            print(f"📄 [Debug] 片段 1 预览: {docs[0].page_content[:100]}...") # 打印前100个字看看
            # print(f"📄 [Debug] 来源: {docs[0].metadata}")
        else:
            print("⚠️ [Debug] 警告：没有检索到任何文档！Context 为空！")
        # ==================== 🛠️ Debug 代码结束 ====================
        
        context_text = "\n\n---\n\n".join([d.page_content for d in docs]) if docs else ""
        
        # 3. 第三步：生成 (Generation)
        history_text = ""
        if chat_history:
            for role, text in chat_history[-6:]:
                history_text += f"{role}: {text}\n"

        full_prompt = self.prompt.format(
            context=context_text, 
            chat_history=history_text, 
            question=question
        )
        
        print("🧠 Gemini 正在生成回答...")
        resp = self.llm.invoke(full_prompt)
        return getattr(resp, "content", str(resp))


# ========================================================
# 3. LocalRAGEngine
# ========================================================
class LocalRAGEngine:
    def __init__(self, k: int = 3):
        if not LOCAL_MODEL_AVAILABLE:
            raise ImportError("inference.py 未找到")
        loader = VectorDBLoader(k=k)
        self.retriever = loader.load_db()
        self.model, self.tokenizer, self.device = load_local_model()
        
        # 即使是 Local 引擎，如果环境允许，最好也用 Gemini 来重写(更准)，这里简化处理
        self.prompt = PromptTemplate(template=MEDICAL_PROMPT_TEMPLATE, input_variables=["context", "chat_history", "question"])

    def answer_question(self, question: str, chat_history: list = None) -> str:
        # 简化的 Local 逻辑
        # 真正跑的时候建议参考上一条回答的 Hybrid 写法
        docs = self.retriever.invoke(question)
        context_text = "\n".join([d.page_content for d in docs]) if docs else ""
        history_text = "\n".join([f"{r}: {t}" for r,t in (chat_history or [])[-2:]])
        full_prompt = self.prompt.format(context=context_text, chat_history=history_text, question=question)
        response = generate_local_response(self.model, self.tokenizer, self.device, full_prompt)
        return response