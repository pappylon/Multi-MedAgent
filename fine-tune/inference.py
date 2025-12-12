import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def load_local_model():
    print(f"⏳ [System]: gpu_med_full_model ...")
    
    model_path = os.path.join(PROJECT_ROOT, "models", "gpu_med_full_model")

    # ====================================================
    # 🧠 智能设备选择逻辑 (关键修改)
    # ====================================================
    if torch.cuda.is_available():
        # 情况 A: Windows 电脑 (有 NVIDIA 显卡)
        device = "cuda"
        torch_dtype = torch.float16 # GPU 上用 fp16 既快又省显存
        print("✅ 检测到 CUDA 设备，启用 GPU 加速模式")
        
    elif torch.backends.mps.is_available():
        # 情况 B: 你的 Mac 电脑
        # 虽然 Mac 有 MPS 加速，但因为模型文件太大 (14GB)，会导致 Buffer 溢出报错
        # 所以针对 Mac，我们强制降级到 CPU
        device = "cpu" 
        torch_dtype = torch.float32 # CPU 用 float32 兼容性最好
        print("⚠️ 检测到 Mac MPS，但模型过大 (14GB+)，强制切换至 CPU 模式以避开 Metal 限制。")
        print("💡 提示：本地推理速度会较慢，这是正常的。")
        
    else:
        # 情况 C: 普通电脑 (无显卡)
        device = "cpu"
        torch_dtype = torch.float32
        print("⚠️ 未检测到 GPU，使用 CPU 模式。")

    try:
        # 加载 Tokenizer
        print(f"📂 加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 加载模型
        print(f"📂 加载模型权重...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,           # 智能分配 (Win->cuda, Mac->cpu)
            torch_dtype=torch_dtype,     # 智能类型 (Win->fp16, Mac->fp32)
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        print("✅ 本地模型加载成功！")
        return model, tokenizer, device

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None
    
def generate_local_response(model, tokenizer, device, prompt_text):
    """生成回答 (增强版)"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3, # 降低温度，让重写更稳定
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # ✅ 增强解析逻辑
    # 1. 尝试按标准格式截取
    if "### Output:" in full_response:
        return full_response.split("### Output:")[-1].strip()
    
    # 2. 如果模型没写 Output 标签，尝试去掉 Prompt 本身
    # (有些模型会把 Prompt 复述一遍)
    if full_response.startswith(prompt_text):
        return full_response[len(prompt_text):].strip()
        
    # 3. 实在没办法，返回原来的全部内容 (总比返回空好)
    return full_response.strip()