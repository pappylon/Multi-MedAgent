import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_DB_DIR", "data/vector_store")
PDF_PATH = os.getenv("PDF_SOURCE_DIR", "data/raw_pdfs")


# ======================================================
# 1. RAG 问答模板 (依赖 Context)
# ======================================================
MEDICAL_PROMPT_TEMPLATE = """### Instruction:
You are a concise medical assistant.
Answer the user's question strictly based on the provided "Context".
If the context does not contain the answer, please first politely indicate that no relevant information was found in external databases, then provide a reference response.
Do not make up information.

Context:
{context}

Chat History:
{chat_history}

Current Question:
{question}

### Output:
"""

# ======================================================
# 2. 问题重写模板 (解决代词指代问题)
# ======================================================
REWRITE_PROMPT_TEMPLATE = """### Instruction:
Rewrite the "Latest Question" into a standalone question using context from "Chat History".
Only return the rewritten question string. No explanations.

Chat History:
{chat_history}

Latest Question:
{question}

### Output:
"""

# ======================================================
# 3. ✅ 新增：纯对话模板 (直连本地模型，不依赖 Context)
# ======================================================
DIRECT_CHAT_TEMPLATE = """You are a helpful and professional medical assistant.
Answer the user's question based on your internal knowledge. 
Be concise, safe, and empathetic.

Chat History:
{chat_history}

Current Question:
{question}

### Output:
"""
