import os
import sys
from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv

# 获取项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 1. 加载环境变量中的 Token
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    if not token:
        print("❌ 错误：未找到 HF_TOKEN。请确保在 .env 文件中配置了 HF_TOKEN=hf_...")
        return

    # 2. 登录 Hugging Face
    print("🔐 正在验证 Hugging Face Token...")
    try:
        login(token=token)
        print("✅ 登录成功！")
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return

    # 3. 配置下载参数
    model_id = "meta-llama/Meta-Llama-3-8B"
    
    # 将模型下载到项目根目录下的 models 文件夹中
    local_dir = os.path.join(PROJECT_ROOT, "models", "Meta-Llama-3-8B")

    print(f"🚀 开始下载模型: {model_id}")
    print(f"📂 保存目标路径: {local_dir}")
    print("⏳ 下载过程可能需要较长时间 (约 15GB)，请保持网络通畅...")

    try:
        # 使用 snapshot_download 直接下载文件到本地
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=token,
            # 优化：只下载 PyTorch 权重和必要配置文件，忽略其他框架的文件以节省空间
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"],
            local_dir_use_symlinks=False # 确保下载的是真实文件而不是链接
        )
        print("\n" + "="*50)
        print(f"🎉 恭喜！模型已成功下载到: {local_dir}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 下载过程中发生错误: {e}")
        if "401" in str(e) or "403" in str(e):
            print("💡 提示：请检查你是否已在 Hugging Face 官网上申请了 Llama 3 的访问权限。")

if __name__ == "__main__":
    main()