import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1") 

CHROMA_PATH = os.getenv("CHROMA_DB_DIR", "data/vector_store")
PDF_PATH = os.getenv("PDF_SOURCE_DIR", "data/raw_pdfs")