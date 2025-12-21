import sys
import os
import inspect # 用于打印模块路径
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

# 导入配置和加载器
from rag.loader import VectorDBLoader
from rag.config import MEDICAL_PROMPT_TEMPLATE, REWRITE_PROMPT_TEMPLATE, DIRECT_CHAT_TEMPLATE

# ========================================================
# 1. 关键修复：强制优先加载 fine-tune 路径
# ========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
fine_tune_dir = os.path.join(project_root, "fine-tune")

# 🔴 关键修改：使用 insert(0) 而不是 append
# 这确保 Python 第一个去 fine-tune 文件夹找 inference.py
if fine_tune_dir not in sys.path:
    sys.path.insert(0, fine_tune_dir)

try:
    import inference # 先导入模块
    from inference import load_local_model, generate_local_response
    
    # 🕵️‍♂️ Debug: 打印到底加载了哪里的 inference 文件
    print(f"✅ 成功导入 inference 模块")
    print(f"   📂 来源路径: {os.path.abspath(inference.__file__)}")
    
    # 二次检查：确保函数不是 None
    if generate_local_response is None:
        raise ImportError("generate_local_response 函数为空 (None)！")
        
    LOCAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入 inference 模块: {e}")
    print(f"   👀 当前 sys.path: {sys.path}")
    LOCAL_MODEL_AVAILABLE = False
except Exception as e:
    print(f"⚠️ 导入 inference 时发生未知错误: {e}")
    LOCAL_MODEL_AVAILABLE = False

# ========================================================
# 2. GeminiRAGEngine
# ========================================================
class GeminiRAGEngine:
    def __init__(self, google_api_key: str, k: int = 5, temperature: float = 0):
        print("🔍 [Gemini] 初始化 RAG 检索器...")
        loader = VectorDBLoader(k=k)
        self.retriever = loader.load_db()

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
        
        self.prompt = PromptTemplate(
            template=MEDICAL_PROMPT_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )
        
        rewrite_prompt_template = PromptTemplate(
            template=REWRITE_PROMPT_TEMPLATE,
            input_variables=["chat_history", "question"]
        )
        self.rewrite_chain = rewrite_prompt_template | self.llm | StrOutputParser()

    def rewrite_query(self, question: str, chat_history: list) -> str:
        if not chat_history:
            return question
        history_text = "\n".join([f"{role}: {text}" for role, text in chat_history[-3:]])
        try:
            new_q = self.rewrite_chain.invoke({"chat_history": history_text, "question": question})
            return new_q.strip()
        except Exception as e:
            print(f"⚠️ [Gemini] 重写失败: {e}")
            return question

    def answer_question(self, question: str, chat_history: list = None) -> str:
        search_query = self.rewrite_query(question, chat_history)
        print(f"🔍 [Gemini] 正在检索: {search_query}")
        docs = self.retriever.invoke(search_query)
        print(f"📄 [Gemini] 检索到了 {len(docs)} 个相关片段")
        
        context_text = "\n\n---\n\n".join([d.page_content for d in docs]) if docs else ""
        history_text = ""
        if chat_history:
            for role, text in chat_history[-6:]:
                history_text += f"{role}: {text}\n"

        full_prompt = self.prompt.format(context=context_text, chat_history=history_text, question=question)
        print("🧠 [Gemini] 正在生成回答...")
        resp = self.llm.invoke(full_prompt)
        return getattr(resp, "content", str(resp))


# ========================================================
# 3. LocalRAGEngine
# ========================================================
class LocalRAGEngine:
    def __init__(self, k: int = 5):
        if not LOCAL_MODEL_AVAILABLE:
            raise ImportError("inference.py not found. Please check the fine-tune directory.")
            
        print("Initializing RAG Retriever......")
        # loader = VectorDBLoader(k=k)
        # self.retriever = loader.load_db()
        
        print("Load local model...")
        self.model, self.tokenizer, self.device = load_local_model()
        
        self.prompt = PromptTemplate(
            template=MEDICAL_PROMPT_TEMPLATE, 
            input_variables=["context", "chat_history", "question"]
        )
        self.rewrite_prompt = PromptTemplate(
            template=REWRITE_PROMPT_TEMPLATE, 
            input_variables=["chat_history", "question"]
        )

    def rewrite_query(self, question: str, chat_history: list) -> str:
        """Rewriting Issues Using Local Models"""
        if not chat_history:
            return question
        
        history_text = "\n".join([f"{role}: {text}" for role, text in chat_history[-3:]])
        full_rewrite_prompt = self.rewrite_prompt.format(chat_history=history_text, question=question)
        
        try:
            print(f"Rewriting the question...")

            new_q = generate_local_response(self.model, self.tokenizer, self.device, full_rewrite_prompt)
            return new_q.strip().split('\n')[0]
        except Exception as e:
            print(f"Rewrite failed: {e}")
            return question

    def answer_question(self, question: str, chat_history: list = None) -> str:
        search_query = self.rewrite_query(question, chat_history)
        
        docs = self.retriever.invoke(search_query)
        
        context_text = "\n\n---\n\n".join([d.page_content for d in docs]) if docs else ""
        
        history_text = ""
        if chat_history:
            for role, text in chat_history[-6:]:
                history_text += f"{role}: {text}\n"

        full_prompt = self.prompt.format(
            context=context_text, 
            chat_history=history_text, 
            question=question
        )
        

        if generate_local_response is None:
            return "Internal error: The generate_local_response function has not been loaded."
            
        response = generate_local_response(self.model, self.tokenizer, self.device, full_prompt)
        return response
    
class LocalLLMEngine:
    def __init__(self):
        if not LOCAL_MODEL_AVAILABLE:
            raise ImportError("inference.py not found. Please check the fine-tune directory.")
            
        print("🚀 [LocalLLM] Initializing Direct Local Model (No RAG)...")
        
        # ❌ 不加载 VectorDBLoader
        # self.retriever = ... (不需要)
        
        print("Load local model...")
        # 复用 inference.py 里的加载函数
        self.model, self.tokenizer, self.device = load_local_model()
        
        # 使用纯对话模板
        self.prompt = PromptTemplate(
            template=DIRECT_CHAT_TEMPLATE, 
            input_variables=["chat_history", "question"]
        )


    def answer_question(self, question: str, chat_history: list = None) -> str:
        """
        直接调用大模型进行回答，不进行检索
        """
        # 1. 格式化历史记录
        history_text = "None"
        total_input = []
        total_input.append({"role": "system", "content": "You are a helpful and professional medical assistant. "
        "Answer the user's question based on your internal knowledge. "
        "Be concise, safe, and empathetic"
        })
        history_list = []
        if chat_history:
            # 拼接最近 6 条记录
            history_text = ""
            for role, text in chat_history[-6:]:
                history_text += f"{role}: {text}\n"
                history_list.append({"role": role, "content":text})

        # total_input.extend(history_list)
        # total_input.append({"role": "user", "content": question})
        # total_input.extend(history_list)
        # 2. 填充 Prompt (注意：这里不需要 context 参数了)
        full_prompt = self.prompt.format(
            chat_history=history_text, 
            question=question
        )
        
        print(f"🤖 [LocalLLM] Generating response for: {question}")

        print("\n*****************" + full_prompt + "******************\n")


        # 3. 检查生成函数是否存在
        if generate_local_response is None:
            return "Internal error: The generate_local_response function has not been loaded."
            
        # 4. 调用 inference.py 进行生成
        # 注意：generate_local_response 会自动加上 <start_of_turn>user 等标签
        response = generate_local_response(self.model, self.tokenizer, self.device, full_prompt)
        
        return response