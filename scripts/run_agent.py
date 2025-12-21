import sys
import os
from pathlib import Path
import textwrap
from dotenv import load_dotenv

# 屏蔽 HuggingFace tokenizer 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# 路径设置：确保能找到 src 目录
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# ✅ 改动 1: 导入本地引擎
from rag.engine import LocalRAGEngine 
# from rag.engine import GeminiRAGEngine

# -------------------------------
# 加载环境变量
# -------------------------------
load_dotenv()
# 虽然本地推理不需要 Google Key，但向量检索 (Embedding) 可能还在用
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# -------------------------------
# CLI 交互逻辑
# -------------------------------
def run_cli(engine):
    chat_history = []
    print("\n🚀 医疗助手 (Local Fine-tuned Mode) 已启动！")
    print("💡 提示：本地模型运行速度取决于你的电脑配置，请耐心等待。\n")

    while True:
        try:
            query = input("👨‍⚕️ 请输入医疗问题： ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit"]:
                print("👋 再见！")
                break

            # print("⏳ 正在检索并调用本地模型生成回答 (可能需要几十秒)...")
            
            # 调用回答
            answer = engine.answer_question(query, chat_history)

            print("\n🤖 微调模型回答：\n")
            print(textwrap.fill(answer, width=80))

            # 更新历史记录
            chat_history.append(("user", query))
            chat_history.append(("assistant", answer))

        except KeyboardInterrupt:
            print("\n\n👋 程序被终止")
            break
        except Exception as e:
            print(f"❌ 运行错误：{e}")


def main():
    # ✅ 改动 2: 初始化本地引擎
    # k=3 是为了限制上下文长度，防止 MacBook Air 内存溢出
    print("⏳ 系统正在初始化，加载模型中...")
    
    try:
        # 这里不需要传入 api_key
        engine = LocalRAGEngine(k=3)
        # engine = GeminiRAGEngine(google_api_key=GOOGLE_API_KEY)
        run_cli(engine)
    except Exception as e:
        print(f"❌ 引擎启动失败: {e}")
        print("💡 请检查 models 文件夹下是否已放入模型文件。")

if __name__ == "__main__":
    main()