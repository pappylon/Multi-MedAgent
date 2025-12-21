import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_DB_DIR", "data/vector_store")
PDF_PATH = os.getenv("PDF_SOURCE_DIR", "data/raw_pdfs")


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

REWRITE_PROMPT_TEMPLATE = """### Instruction:
Rewrite the "Latest Question" into a standalone question using context from "Chat History".
Only return the rewritten question string. No explanations.

Chat History:
{chat_history}

Latest Question:
{question}

### Output:
"""

