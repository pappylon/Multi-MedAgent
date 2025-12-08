# loader.py — 加载向量数据库

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_PATH

class VectorDBLoader:
    """加载 Chroma 向量数据库并提供检索器"""

    def __init__(self, persist_dir=CHROMA_PATH, embedding_model_name="all-MiniLM-L6-v2", k=5):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model_name
        self.k = k
        self.db = None
        self.retriever = None
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def load_db(self):
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(f"Chroma 数据库目录不存在: {self.persist_dir}")
        self.db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={"k": self.k})
        return self.retriever
