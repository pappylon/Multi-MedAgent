from collections.abc import Callable
from typing import Dict
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig, clone_chat_template, setup_chat_format
import random

from loguru import logger
import logging


#huggingface拉取模型的token
def init_huggingface_token():
    TOKEN = "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"
    from huggingface_hub import login
    login(token=TOKEN)

import math
#对数据进行采样
def dataset_sample(dataset, sample_ratio: int=0.5, size: int=2000) -> Dataset:
    random.seed(123)
    all_indices = list(range(len(dataset)))
    if sample_ratio:
        _len = len(all_indices)
        sample_size = math.ceil(_len * sample_ratio)
    else:
        sample_size = size
    sampled_indices = random.sample(all_indices, sample_size)
    sampled_dataset = dataset.select(sampled_indices)
    logger.info(f"采样后的数据集长度为: {len(sampled_dataset)}")
    return sampled_dataset


def formatting_prompts_func_MedQuAd(examples, tokenizer):
    '''
    1w6的medquad数据集处理
    :param examples:
    :param tokenizer:
    :return:
    '''
    texts = []

    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    CUT_SIZE = 4096

    # if not all(k in examples and examples[k] is not None for k in ["qtype", "Question", "Answer"]):
    #     logger.debug(f"数据集 medquad 部分内容有缺失")
    for instruction, input_data, output in zip(examples["qtype"], examples["Question"], examples["Answer"]):
        message = []
        if instruction and instruction.strip():
            message.append({"role": SYSTEM_ROLE, "content": f"question category: {instruction}"})

        if input_data and input_data.strip():
            message.append({"role": USER_ROLE, "content": input_data})

        if output and output.strip():
            message.append({"role": ASSISTANT_ROLE, "content": output})

        formatted_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
        )

        if len(formatted_text) <= CUT_SIZE:
            texts.append(formatted_text)
    return {"text": texts}


# 进行数据格式化
def formatting_prompts_func_MedicalQa(examples, tokenizer):
    '''
    23w的medical-qa数据集的prompt处理
    :param examples:
    :param tokenizer:
    :return:
    '''
    messages = []

    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    CUT_SIZE = 4096
    # if not all(k in examples and examples[k] is not None for k in ["question", "complex_cot", "response"]):
    #     logger.debug(f"数据集 medical-qa 部分内容有缺失")

    for instruction, input_data, output in zip(examples["instruction"], examples["input"], examples["output"]):
        message = []
        if instruction and instruction.strip():
            message.append({"role": SYSTEM_ROLE, "content": f"question category: {instruction}"})

        if input_data and input_data.strip():
            message.append({"role": USER_ROLE, "content": input_data})

        if output and output.strip():
            message.append({"role": ASSISTANT_ROLE, "content": output})
        else:
            continue

        formatted_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,  # 如若还需以新的prompt进行设置为true
        )
        if len(formatted_text) <= CUT_SIZE:
            messages.append(message)

    return {"messages": messages}


def formatting_prompts_func_MedQuAd_raw(examples, tokenizer):
    '''
    1w6的medquad数据集处理
    :param examples:
    :param tokenizer:
    :return:
    '''
    messages = []

    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    CUT_SIZE = 4096

    # if not all(k in examples and examples[k] is not None for k in ["qtype", "Question", "Answer"]):
    #     logger.debug(f"数据集 medquad 部分内容有缺失")

    for instruction, input_data, output in zip(examples["qtype"], examples["Question"], examples["Answer"]):
        message = []
        if instruction and instruction.strip():
            message.append({"role": SYSTEM_ROLE, "content": f"question category: {instruction}"})

        if input_data and input_data.strip():
            message.append({"role": USER_ROLE, "content": input_data})

        if output and output.strip():
            message.append({"role": ASSISTANT_ROLE, "content": output})

        formatted_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
        )
        if len(formatted_text) <= CUT_SIZE:
            messages.append(message)

    return {"messages": messages}


def formatting_prompts_func_MedicalO1(examples, tokenizer):
    messages = []
    token_len = []

    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    CUT_SIZE = 4096

    # if not all(k in examples and examples[k] is not None for k in ["Question", "Complex_CoT", "Response"]):
    #     logger.debug(f"数据集 medical-o1 部分内容有缺失")
    for question, cot, answer in zip(examples["Question"], examples["Complex_CoT"], examples["Response"]):
        message = []
        cot = " ".join(cot.split("\n\n"))
        assistant_content = f"<think>{cot}</think>\n\n{answer}"

        message.append({"role": SYSTEM_ROLE, "content": "You are a professional medical expert. "
                                                        "Prior to answering the question, "
                                                        "please first present a detailed "
                                                        "clinical reasoning process (Chain of Thought), "
                                                        "and then provide the final answer."})

        if question and question.strip():
            message.append({"role": USER_ROLE, "content": question})

        if cot and cot.strip() and answer and answer.strip():
            message.append({"role": ASSISTANT_ROLE, "content": assistant_content})

        formatted_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
        )

        if len(formatted_text) <= CUT_SIZE:
            messages.append(message)
        # # 标准的Instruction Tuning Prompt格式
        # if input_data and input_data.strip():
        #     # 有输入时的 Prompt 模板
        #     prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_data}\n### Output:\n{output}\n"
        # else:
        #     # 无输入时的 Prompt 模板
        #     prompt = f"### Instruction:\n{instruction}\n### Output:\n{output}\n"
        # texts.append(prompt)

    return {"messages": messages}


def formatting_prompts_func_PubMedQa(examples, tokenizer):
    messages = []

    CUT_SIZE = 4096
    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"

    # if not all(k in examples and examples[k] is not None for k in
    #            ["question", "context", "long_answer", "final_decision"]):
    #     logger.debug(f"数据集 pubmedqa 部分内容有缺失")

    for question, context, answer, decision in zip(examples["question"], examples["context"], examples["long_answer"],
                                                   examples["final_decision"]):
        message = []
        structured_bg = ""
        for label, text in zip(context['labels'], context['contexts']):
            # 加上清晰的标题分隔符，帮助模型识别逻辑段落
            structured_bg += f"### {label}:\n{text}\n"
        keywords = ", ".join(context['meshes'])
        sys_prompt = ("You are a clinical medicine expert. "
                      "Please answer the question based on the provided research abstract "
                      "(Background) and keywords. "
                      "The analysis must be accurate and logically rigorous, "
                      "and a final interpretation must be given.")
        message.append({"role": SYSTEM_ROLE, "content": sys_prompt})
        user_content = (
            f"【research background】\n:{structured_bg}\n"
            f"【keywords】: {keywords}\n"
            f"【question】: {question}\n"
            f"Please conduct a detailed analysis of the study's results "
            f"based on the aforementioned evidence and provide a final conclusion."
        )
        if context and question and question.strip():
            message.append({"role": USER_ROLE, "content": user_content})
        assistant_content = (
            f"【analysis】: {answer}\n"
            f"【final result】：[{decision}]"
        )

        if answer and decision and answer.strip() and decision.strip():
            message.append({"role": ASSISTANT_ROLE, "content": assistant_content})

        formatted_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
        )

        if len(formatted_text) <= CUT_SIZE:
            messages.append(message)

    return {"messages": messages}

def configure_lora_config(model_name:str) -> LoraConfig:
    name_target_projection = {
        "Llama":['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        "Qwen":['c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj'],
        "Falcon":['query_key_value', 'dense', 'dense_h_to_4h','dense_4h_to_h'],
        "ChatGLM":['query_key_value','dense','dense_h_to_4h','dense_4h_to_h'],
        "default":['q_proj', 'k_proj', 'v_proj'],
    }

    for key in name_target_projection.keys():
        if key.lower() in model_name.lower():
            logger.info(f"find the target projection for {key}")
            target_module = name_target_projection[key]
            break
        else:
            logger.error(f"can not find model---{model_name}'s target_module parameters!")
            logger.info(f"model---{model_name}utilizing default Q-K-V param")
            target_module = name_target_projection["default"]

    peft_config = LoraConfig(
        lora_alpha=lora_param["LORA_ALPHA"],
        lora_dropout=lora_param['LORA_DROPOUT'],
        r=lora_param['LORA_R'],
        bias="none",  # lora_only,
        task_type="CAUSAL_LM",
        target_modules=target_module,
    )
    logger.info(f"lora config loading success!")
    return peft_config

def get_training_device():
    device_count = torch.cuda.device_count()
    if device_count > 0:
        device = torch.device("cuda")
        print(f"using device: {device}; num: {device_count}")
    else:
        device = torch.device("cpu")
        print(f"使用设备: {device}")
    return device

def calculate_flops(batch_size_per_train, max_seq_legnth) -> int:
    model_flops = (
        model.floating_point_ops(
            {
                "input_ids": torch.zeros(
                    (batch_size_per_train, max_seq_legnth),
                )
            }
        )
    ) * sft_config.gradient_accumulation_steps
    return model_flops

def load_total_dataset(dataset_path_func_map: Dict[str, Callable], tokenizer):
    train_dataset = None
    test_dataset = None

    for file_path, func in dataset_path_func_map.items():
        dataset = load_dataset("json", data_files=file_path, split="train", cache_dir=CACHE_DIR)
        tmp = dataset.map(
            func,
            fn_kwargs={"tokenizer": tokenizer},
            batched=True,
            batch_size=2000,
            remove_columns=dataset.column_names
        )
        splits = tmp.train_test_split(test_size=0.1, shuffle=True, seed=42)
        if train_dataset is None:
            train_dataset = splits["train"]
        else:
            train_dataset = concatenate_datasets([train_dataset, splits["train"]])
        if test_dataset is None:
            test_dataset = splits["test"]
        else:
            test_dataset = concatenate_datasets([test_dataset, splits["test"]])

    test_dataset.shuffle(seed=42)
    train_dataset.shuffle(seed=42)
    logger.info(f"eval dataset--size{len(test_dataset)} **** "
                f"train dataset--size{len(train_dataset)} loading complete!")
    return train_dataset, test_dataset

def load_tokenizer(model_name, cache_dir):
    # load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    # set padding_side='right' to avoid efficiency problem，and eos_token for padding
    if 'llama' in MODEL_NAME.lower():
        tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'left'
    if "generation" not in tokenizer.chat_template:
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

    logger.info(f"tokenizer name: {tokenizer.__class__.__name__} loading complete!")
    return tokenizer

def load_token_param():
    return

def resume_training():
    return

# basic param set
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = r"D:\huggingface_cache"
OUTPUT_DIR = "./med_lora_results"
LORA_WEIGHT_PATH = "../gpu_med_lora_results"  # 上一步保存 LoRA 权重的本地路径

dataset_path_func_map = {
    '../datasets/MedQuad-MedicalQnADataset/'
    'json/MedQuad-MedicalQnADataset.jsonl': formatting_prompts_func_MedQuAd_raw,
    '../datasets/medical-qa-datasets-all-processed/'
    'json/medical-qa-datasets.jsonl': formatting_prompts_func_MedicalQa,
    '../datasets/medical-o1-reasoning-SFT-en/'
    'json/medical-o1-reasoning-SFT.jsonl': formatting_prompts_func_MedicalO1,
    '../datasets/PubMedQA-pqa_artificial/'
    'json/PubMedQA.jsonl': formatting_prompts_func_PubMedQa,
}

# 4-bit quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    global_config = None

    lora_param = {
        "LORA_R": 8,  # LoRA rank
        "LORA_DROPOUT": 0.1,  # LoRA dropout rate
        "LORA_ALPHA": 16,  # LoRA scaling factor
    }
    train_param = {
        "BATCH_SIZE": 1,
        "GRADIENT_ACCUMULATION_STEPS": 4,
        "LEARNING_RATE": 2e-4,
        "MAX_SEQ_LENGTH": 4096,
        "NUM_TRAIN_EPOCHS": 10,
        "FP16": False,
        "BF16" : torch.cuda.is_available(),
        "PER_DEVICE_TRAIN_BATCH_SIZE": 1,
        "LOGGING_STEPS": 10,
        "SAVE_STEPS": 1000,
    }

    # --- 2. 模型和 Tokenizer 加载 (4-bit 量化) ---

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )
    # model.to(get_training_device())
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    logger.info(f"Model---{model._get_name()} loading complete!")


    # load Tokenizer
    tokenizer = load_tokenizer(MODEL_NAME, CACHE_DIR)

    # --- 3. 加载数据 ---
    train_total, test_total = load_total_dataset(dataset_path_func_map, tokenizer)

    # --- 4. LoRA 配置 ---
    peft_config = configure_lora_config(model_name=MODEL_NAME)

    evaluations_metric_list = [
        'eval_accuracy',
        'eval_f1',
        ##############
        'eval_rouge-l',
        'eval_bleu',
        'eval_loss',
    ]

    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # --- 5. SFTConfig param ---
    sft_config = SFTConfig(
        packing=True, #true来提高训练速度
        max_length=train_param['MAX_SEQ_LENGTH'],
        dataset_text_field="text",
        # completion_only_loss=True, # for prompt completion
        assistant_only_loss=True, # for assistant system user
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=train_param['PER_DEVICE_TRAIN_BATCH_SIZE'],
        gradient_accumulation_steps=train_param['GRADIENT_ACCUMULATION_STEPS'],
        optim="paged_adamw_8bit",  # AdamW
        logging_steps=train_param['LOGGING_STEPS'],
        learning_rate=train_param['LEARNING_RATE'],
        fp16=train_param['FP16'],
        bf16=train_param['BF16'],
        max_steps=-1,  #
        # dataloader_drop_last=True, # distributed training true
        num_train_epochs=train_param['NUM_TRAIN_EPOCHS'],
        save_steps=train_param['SAVE_STEPS'],
        save_strategy="steps",
        logging_strategy="steps",
        warmup_ratio=0.03,  # 0.03-0.1
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,  # true reduce ram, lower speed
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        save_total_limit=3,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=1000,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        loss_type='dft',
        dataloader_num_workers=4,
    )

    logger.info(f"Memory footprint: {model.get_memory_footprint() / 1e9} GB")
    logger.info(f"Flops result: {calculate_flops(train_param['PER_DEVICE_TRAIN_BATCH_SIZE'], train_param['MAX_SEQ_LENGTH'])}")

    # print(train_totle['prompt'][0])
    # print(test_totle['prompt'][0])

    # --- 7. SFT Trainer 启动 ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_total,
        eval_dataset=test_total,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    logger.info(f"SFTTrainer load success!")

    logger.info(f"===============start training model===============")
    trainer.train()
    logger.info(f"===============model training complete===============")

    # --- 8. 保存 LoRA 适配器 ---
    logger.info(f"保存LoRA适配器及tokenizer!")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"保存完成!")

    # arc评估

    # --- 9. 合并模型（可选，但推荐用于部署）---
    # 为了部署方便，可以将 LoRA 权重与原始模型合并
    # base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    # merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    # merged_model = merged_model.merge_and_unload()
    # merged_model.save_pretrained("./merged_medical_model")
    # tokenizer.save_pretrained("./merged_medical_model")