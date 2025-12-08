import sys
import os
import textwrap

# 屏蔽 macOS 上 Tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.engine import GeminiRAGEngine

GOOGLE_API_KEY = "AIzaSyAu2CESjHm1fGX7PJ_E6Embl68NjU2dhNs"

def main():
    print("🚀 启动医疗 Agent (Medical Mode - Gemini)")
    engine = GeminiRAGEngine(google_api_key=GOOGLE_API_KEY)

    print("✅ 医疗助手已启动！输入 quit 或 exit 退出。\n")

    while True:
        query = input("👨‍⚕️ 请输入医疗问题： ").strip()
        if not query:
            continue
        if query.lower() in ["quit", "exit"]:
            print("👋 再见！")
            break

        print("⏳ 正在调用 Gemini 生成回答...")
        try:
            answer = engine.answer_question(query)
            print("\n🤖 Gemini 回答：\n")
            print(textwrap.fill(answer, width=80))
        except Exception as e:
            print(f"❌ 错误：{e}")

if __name__ == "__main__":
    main()
