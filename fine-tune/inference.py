import os
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def load_local_model():
    print(f"⏳ [System]: 正在初始化模型路径...")
    
    # 1. 定义路径
    # 基础模型路径
    base_model_path = os.path.join(PROJECT_ROOT, "models", "gpu_med_full_model")

    # LoRA 适配器路径
    adapter_path = os.path.join(base_model_path, "lora_medquad_1_epoch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 定义 4-bit 量化配置 (QLoRA 核心)
    # 这能大幅降低显存占用 (16GB -> 6GB)，并解决 OOM 崩溃问题
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # 显卡如果较老(如10系列)，可能需要改为 torch.float16
        bnb_4bit_use_double_quant=False,
    )

    try:
        # 2. 加载 Tokenizer
        print(f"📂 加载 Tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        # if 'llama' in MODEL_NAME.lower():
        # tokenizer.padding_side = "right"
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = 'left'
        if tokenizer.chat_template and "generation" not in tokenizer.chat_template:
            tokenizer.chat_template(
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
                "{% if message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% generation %}"
                "{{ message['content'] | trim + '<|eot_id|>' }}"
                "{% endgeneration %}"
                "{% else %}"
                "{{ content }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            )

        # 3. 加载基础模型 (应用 4-bit 量化)
        print(f"📂 加载基础模型 (4-bit Quantization)...")
        
        # Windows 兼容性处理：
        # 如果是 CPU 模式，不能用 4-bit 量化；如果是 GPU，尝试加载
        if device == "cuda":
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    quantization_config=bnb_config, # ✅ 应用队友的量化配置
                    device_map="auto",              # 让 accelerate 自动分配设备
                    trust_remote_code=True
                )
            except ImportError:
                print("⚠️ 未检测到 bitsandbytes 库或不支持 4-bit，回退到 FP16 模式...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
        else:
            # CPU 模式
            print("⚠️ 使用 CPU 模式 (速度较慢)...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )

        # 4. 加载 LoRA 微调参数
        if os.path.exists(adapter_path):
            print(f"🔗 正在挂载 LoRA 微调参数: {os.path.basename(adapter_path)} ...")
            try:
                model = PeftModel.from_pretrained(model, adapter_path)
                print("✅ LoRA 微调参数加载成功！(医疗模式已激活)")
            except Exception as e:
                print(f"⚠️ LoRA 加载报错: {e}")
        else:
            print(f"\n❌ [警告] 找不到 LoRA 路径: {adapter_path}，将仅使用基础模型。")

        return model, tokenizer, device

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None
    
def generate_local_response(model, tokenizer, device, formatted_prompt_text):
    """
    接收已经填充好的完整 Prompt，流式输出后续内容
    """
    try:
        # Llama-3 官方格式封装 (可选，取决于你微调时有没有加这个)
        # 如果你微调时直接用的 ### Instruction 格式，可以把下面这行 f-string 去掉，直接用 formatted_prompt_text
        final_input = f"<start_of_turn>user\n{formatted_prompt_text}<end_of_turn>\n<start_of_turn>model\n"

        inputs = tokenizer(final_input, return_tensors="pt").to(model.device)
        
        # skip_prompt=True
        # 它会自动计算输入有多长，输出时只显示模型新生成的部分
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            max_new_tokens=128, 
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        print("🤖 [AI]: ", end="", flush=True)
        
        full_response = ""
        

        for new_text in streamer:
            clean_text = new_text.replace("### Output:", "").replace("###", "").strip()
            
            if not clean_text:
                continue
                
            print(new_text, end="", flush=True) # 打印原始流式文本保持流畅
            full_response += new_text

        print() # 换行
        return full_response.strip()

    except Exception as e:
        print(f"Error: {e}")
        return ""

if __name__ == "__main__":
    m, t, d = load_local_model()
    if m:
        generate_local_response(m, t, d, "I have a headache")