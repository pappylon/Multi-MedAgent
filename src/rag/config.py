import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_DB_DIR", "data/vector_store")
PDF_PATH = os.getenv("PDF_SOURCE_DIR", "data/raw_pdfs")

# 必须匹配 model.py 里的 formatting_prompts_func 格式 Instruction (指令) + Input (上下文) -> Output (回答)
MEDICAL_PROMPT_TEMPLATE = """### Instruction:
You are a helpful medical expert. 
Answer the user's question based on the provided context. 
If the context contains relevant information, use it to form a comprehensive answer.
If the context doesn't explicitly state the answer, try to infer it from the context, or state what is mentioned about the topic.

Current Question: {question}

Conversation History:
{chat_history}

### Input:
{context}

### Output:
"""

# 专门用于把 "它" (it) 替换成具体的名词
REWRITE_PROMPT_TEMPLATE = """### Instruction:
Your task is to rewrite the "Latest Question" into a standalone question that clearly includes the context from the "Chat History".
Replace pronouns like "it", "they", "this" with specific nouns from the history.
Do NOT answer the question. Just output the rewritten question string.

Examples:
History: User: I have a headache. AI: You can take paracetamol.
Latest Question: What is the dosage for it?
Rewritten Question: What is the dosage for paracetamol?

History: User: Tell me about Diabetes. AI: It is a chronic disease...
Latest Question: How to treat it?
Rewritten Question: How to treat Diabetes?

Chat History:
{chat_history}

Latest Question:
{question}

### Output:
"""