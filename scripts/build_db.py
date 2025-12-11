import sys
import os

# 1. 路径修正：确保能找到 src 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. 导入加载器 (使用新版导入路径，修复 ModuleNotFoundError)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 3. 导入向量模型 (换成免费的 HuggingFace，修复 404/额度问题)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 4. 导入配置
from src.config import CHROMA_PATH, PDF_PATH

def build():
    print(f"🔄 正在从 {PDF_PATH} 读取数据...")
    
    # 检查文件夹是否存在
    if not os.path.exists(PDF_PATH):
        os.makedirs(PDF_PATH)
        print(f"❌ 错误: 文件夹 {PDF_PATH} 不存在。请创建它并放入 PDF 文件。")
        return

    # 加载 PDF
    loader = DirectoryLoader(PDF_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("⚠️ 警告: 文件夹里没有找到 PDF 文件，请检查路径。")
        return
    
    print(f"✅ 成功加载 {len(documents)} 页文档。")

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  文档已切分为 {len(chunks)} 个片段。")
    
    # 入库 (使用本地免费模型)
    print(f"💾 正在使用本地模型(HuggingFace)生成向量... (第一次运行需下载模型，请耐心等待)")
    
    # ✅ 这里改用了本地模型，完全免费，不需要 API Key
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 存入 Chroma
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    
    print(f"🎉 数据库构建完成！位置: {CHROMA_PATH}")

if __name__ == "__main__":
    build()