import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- 1. 配置参数 ---
# 您需要根据您选择的模型替换 MODEL_NAME
# 垂直开源模型示例：'boda-hi/Llama-2-7b-chat-med-sft' (医疗LoRA模型)
# 通用开源模型示例：'meta-llama/Llama-2-7b-hf' 或 'Qwen/Qwen-7B'
MODEL_NAME = "your-medical-llama-or-qwen-model-path"
DATASET_PATH = "med_sft_data.jsonl"  # 您的医疗指令数据路径
OUTPUT_DIR = "./med_lora_results"

# LoRA 配置
LORA_R = 64              # LoRA 秩
LORA_ALPHA = 16          # LoRA scaling factor
LORA_DROPOUT = 0.1       # LoRA dropout rate

# 训练配置
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
NUM_TRAIN_EPOCHS = 3
FP16 = not torch.cuda.is_available() # 如果没有CUDA，设为False
PER_DEVICE_TRAIN_BATCH_SIZE = 4
LOGGING_STEPS = 10
SAVE_STEPS = 50

# --- 2. 加载数据 ---
# TRL 库推荐使用这个格式化函数，将指令和响应合并成一个训练序列
def formatting_prompts_func(examples):
    texts = []
    for instruction, input_data, output in zip(examples['instruction'], examples['input'], examples['output']):
        # 标准的Instruction Tuning Prompt格式
        if input_data and input_data.strip():
            # 有输入时的 Prompt 模板
            prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_data}\n### Output:\n{output}"
        else:
            # 无输入时的 Prompt 模板
            prompt = f"### Instruction:\n{instruction}\n### Output:\n{output}"
        texts.append(prompt)
    return {"text": texts}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# --- 3. 模型和 Tokenizer 加载 (4-bit 量化) ---

# 4-bit 量化配置 (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
# 准备模型进行 k-bit 训练（重要步骤）
model = prepare_model_for_kbit_training(model)

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# 对于某些模型（如 Llama），需要设置 padding_side='right' 以避免训练时的性能问题
if 'llama' in MODEL_NAME.lower():
    tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

# --- 4. LoRA 配置 ---
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM", # 适用于生成任务
    # 针对不同模型选择不同的目标模块，通常是查询（query）、键（key）、值（value）矩阵
    # Llama 目标模块: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    # Qwen 目标模块: ['c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

# --- 5. 训练参数配置 ---
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="paged_adamw_8bit", # 内存优化后的 AdamW
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=FP16,
    max_steps=-1, # 设置为-1则根据num_train_epochs来计算总步数
    num_train_epochs=NUM_TRAIN_EPOCHS,
    save_steps=SAVE_STEPS,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# --- 6. SFT Trainer 启动 ---
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False, # 医疗文本长度不均，通常不启用 Packing
    formatting_func=formatting_prompts_func, # 使用自定义的格式化函数
)

# 开始训练
trainer.train()

# --- 7. 保存 LoRA 适配器 ---
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- 8. 合并模型（可选，但推荐用于部署）---
# 为了部署方便，可以将 LoRA 权重与原始模型合并
# base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
# merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
# merged_model = merged_model.merge_and_unload()
# merged_model.save_pretrained("./merged_medical_model")
# tokenizer.save_pretrained("./merged_medical_model")