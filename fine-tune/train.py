import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer # 使用 SFTTrainer 可以简化监督微调过程

# 1. 定义模型和分词器
model_id = "meta-llama/Meta-Llama-3-8B" # 选用 Llama 3 8B 模型

# 2. 配置 4-bit 量化 (QLoRA 的基础)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16 # 使用 bfloat16 进行计算，加速且保持精度
)

# 3. 加载模型和分词器
# 确保在加载模型时应用量化配置
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # 自动分配到 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 确保设置 pad token，这对于训练非常重要
tokenizer.pad_token = tokenizer.eos_token

# 4. 配置 LoRA (Parameter-Efficient Fine-Tuning)
#
lora_config = LoraConfig(
    r=8, # LoRA 秩，通常 8, 16, 32, 64
    lora_alpha=16, # LoRA 缩放因子
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM", # 适用于生成任务
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # 针对 Llama/Mistral 结构的关键模块
    ]
)

# 5. 应用 LoRA 配置到量化模型
model = get_peft_model(model, lora_config)
# 打印出可训练参数量，你会发现它比总参数量少得多
model.print_trainable_parameters()

# 6. 准备数据 (假设你已经有一个名为 'train_dataset' 的数据集)
# from datasets import load_dataset
# train_dataset = load_dataset("json", data_files="your_data.json", split="train")

# 7. 定义训练参数
training_args = TrainingArguments(
    output_dir="./llama3_8b_finetune_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit", # 适用于 QLoRA 的优化器
    lr_scheduler_type="cosine",
    bf16=True, # 启用 bfloat16
)

# 8. 初始化 SFTTrainer (Supervised Fine-Tuning Trainer)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # 替换为你的数据集变量
    tokenizer=tokenizer,
    peft_config=lora_config,
    max_seq_length=1024, # 序列最大长度
    dataset_text_field="text", # 数据集中包含输入和输出的字段名
)

# 9. 开始训练
trainer.train()

# 10. 保存 LoRA 适配器权重
trainer.model.save_pretrained(training_args.output_dir)
# 注意：这里只保存了小的 LoRA 权重，而不是整个大模型