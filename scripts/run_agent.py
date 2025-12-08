import sys
import os
import textwrap
from dotenv import load_dotenv

# 屏蔽 macOS 上 Tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 确保能找到 src 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.engine import GeminiRAGEngine

load_dotenv()

# 从环境变量获取 Key，而不是硬编码
# 确保你的 .env 文件里有一行：GOOGLE_API_KEY=AIzaSy...
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def main():
    # 新增：安全检查
    if not GOOGLE_API_KEY:
        print("❌ 错误：未找到 GOOGLE_API_KEY 环境变量。")
        print("💡 请检查你的 .env 文件，确保包含 'GOOGLE_API_KEY=你的密钥'")
        return

    print("🚀 启动医疗 Agent (Medical Mode - Gemini)")
    
    # 初始化引擎
    engine = GeminiRAGEngine(google_api_key=GOOGLE_API_KEY)

    # ✅ 保留：初始化对话历史列表
    chat_history = []

    print("✅ 医疗助手已启动！输入 quit 或 exit 退出。\n")

    while True:
        try:
            # 使用 input 获取用户输入
            query = input("👨‍⚕️ 请输入医疗问题： ").strip()
            
            if not query:
                continue
            if query.lower() in ["quit", "exit"]:
                print("👋 再见！")
                break

            print("⏳ 正在调用 Gemini 生成回答...")
            
            # ✅ 保留：传入 chat_history
            answer = engine.answer_question(query, chat_history)
            
            print("\n🤖 Gemini 回答：\n")
            print(textwrap.fill(answer, width=80))
            
            # ✅ 保留：更新历史记录
            chat_history.append(("User", query))
            chat_history.append(("AI", answer))

        except KeyboardInterrupt:
            # 捕获 Ctrl+C 中断
            print("\n\n👋 程序被强制终止")
            break
        except Exception as e:
            print(f"❌ 错误：{e}")

if __name__ == "__main__":
    main()