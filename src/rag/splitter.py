# splitter.py — 文档切分工具

from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    """文本切分器，用于新文档或增量更新"""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)
