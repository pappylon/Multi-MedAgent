import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_DB_DIR", "data/vector_store")
PDF_PATH = os.getenv("PDF_SOURCE_DIR", "data/raw_pdfs")

# 必须匹配 model.py 里的 formatting_prompts_func 格式 Instruction (指令) + Input (上下文) -> Output (回答)

MEDICAL_PROMPT_TEMPLATE = """### Instruction:
You are a professional medical AI assistant. Answer the user's question strictly based on the provided context below.
If the answer is not in the context, say "I don't know".

Current Question: {question}

Conversation History:
{chat_history}

### Input:
{context}

### Output:
"""