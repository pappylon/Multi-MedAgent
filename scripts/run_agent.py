import sys
import os
from pathlib import Path
import textwrap
from dotenv import load_dotenv

# 屏蔽 HuggingFace tokenizer 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# 项目路径设置（从 scripts 回到项目根目录）
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # Multi-MedAgent/
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from rag.engine import GeminiRAGEngine  # type: ignore

# -------------------------------
# 加载环境变量
# -------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# -------------------------------
# CLI 模式
# -------------------------------
def run_cli(engine):
    chat_history = []
    print("🚀 医疗助手 CLI 已启动！输入 quit 或 exit 退出。\n")

    while True:
        try:
            query = input("👨‍⚕️ 请输入医疗问题： ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit"]:
                print("👋 再见！")
                break

            print("⏳ 正在调用 Gemini 生成回答...")
            answer = engine.answer_question(query, chat_history)

            print("\n🤖 Gemini 回答：\n")
            print(textwrap.fill(answer, width=80))

            chat_history.append(("User", query))
            chat_history.append(("AI", answer))

        except KeyboardInterrupt:
            print("\n\n👋 程序被终止")
            break
        except Exception as e:
            print(f"❌ 错误：{e}")


def main():
    if not GOOGLE_API_KEY:
        print("❌ 未找到 GOOGLE_API_KEY，请在 .env 中配置。")
        return

    engine = GeminiRAGEngine(google_api_key=GOOGLE_API_KEY)
    run_cli(engine)


if __name__ == "__main__":
    main()

