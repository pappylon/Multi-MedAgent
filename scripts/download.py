import os
# 1. 强制设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print("🚀 开始下载模型，请稍候...")

# 2. 下载模型到当前目录下的 models 文件夹
try:
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir="./models/all-MiniLM-L6-v2",  # 下载到这里
        local_dir_use_symlinks=False      # Windows 必须设置这个为 False
    )
    print("✅ 下载成功！模型保存在：./models/all-MiniLM-L6-v2")
except Exception as e:
    print(f"❌ 下载失败: {e}")