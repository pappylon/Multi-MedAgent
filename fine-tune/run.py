import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
# PeftModel 是 PEFT 库中用于挂载适配器的核心类

# 1. 定义基础模型和 LoRA 适配器权重保存的路径
base_model_id = "meta-llama/Meta-Llama-3-8B" # 原始的基础模型 ID
lora_weights_path = "./llama3_8b_finetune_results" # 上一步保存 LoRA 权重的本地路径

# 2. 加载基础模型和 Tokenizer
# 注意：加载基础模型时，如果训练时使用了 4bit 量化（QLoRA），推理时最好也使用相同的量化配置，以确保内存和兼容性。
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 确保量化配置与训练时一致，特别是当训练使用的是 QLoRA 时
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 强制加载基础模型到 CPU/GPU，并应用量化配置
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config, # 应用量化配置
    device_map="auto",
    torch_dtype=torch.bfloat16 # 配合计算 dtype
)

# 3. 挂载 LoRA 权重到基础模型
#
model = PeftModel.from_pretrained(
    base_model,
    lora_weights_path,
)

# 4. 模型推理示例
prompt = "请用一句古诗来描述你眼中的大海。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# 此时，变量 'model' 就是一个集成了 LoRA 微调效果的大模型。