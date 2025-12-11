import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def load_local_model():
    print(f"⏳ [System] : gpu_med_full_model ...")

    model_path = os.path.join(PROJECT_ROOT, "models", "gpu_med_full_model")

    # 1. 硬件检测 (适配 Mac 和 Windows)
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ 检测到 CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = "mps" # Mac M1/M2/M3 芯片加速
        print("✅ 检测到 MPS (Mac GPU)")
    else:
        device = "cpu"
        print("⚠️ 未检测到 GPU，将使用 CPU 模式 (速度较慢)")

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型文件夹，请检查路径: {model_path}")

        # 2. 加载分词器 (直接从本地加载)
        print(f"📂 加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 补全设置 (防止报错)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 3. 加载模型 (直接加载完整版)
        print(f"📂 加载模型权重 (可能需要几分钟)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True
        )

        print("✅ 本地模型加载成功！")
        return model, tokenizer, device

    except Exception as e:
        print(f"❌ 模型加载严重失败: {e}")
        # 如果加载失败，为了不让程序崩溃，我们返回 None
        return None, None, None

def generate_local_response(model, tokenizer, device, prompt_text):
    """生成回答"""
    # 确保输入数据也在正确的设备上
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # 稍微调大一点，允许它多说点
            do_sample=True,      # 启用采样，让回答更自然
            temperature=0.5,     # 温度：越低越严谨，越高越有创造力
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 清洗数据：只保留 ### Output: 之后的内容
    if "### Output:" in full_response:
        return full_response.split("### Output:")[-1].strip()
    
    # 如果模型没有严格遵守 Output 格式，尝试去掉 Prompt 部分
    # (简单的字符串去重)
    if full_response.startswith(prompt_text):
        return full_response[len(prompt_text):].strip()

    return full_response