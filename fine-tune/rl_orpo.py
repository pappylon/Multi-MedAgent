import math
from collections.abc import Callable
from typing import Dict
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig, clone_chat_template, setup_chat_format, ORPOConfig, ORPOTrainer
import random

from loguru import logger

from model import (
    load_tokenizer,
    bnb_config,
    dataset_sample,
)

def generate_rl_dataset_usmle(examples, tokenizer):

    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    total_chosen = []
    total_rejected = []
    for question, answer, options, answer_id in zip(
        examples['question'],
        examples['answer'],
        examples['options'],
        examples['answer_idx'],
    ):
        chosen = []
        rejected = []
        if question and question.strip():
            chosen.append({"role": USER_ROLE, "content": question})
            rejected.append({"role": USER_ROLE, "content": question})
        else:
            continue
        if answer and answer.strip():
            chosen.append({"role": ASSISTANT_ROLE, "content": "The answer is " + answer})
        else:
            continue
        if options:
            # random select one
            rejected.append({"role": ASSISTANT_ROLE, "content": "The answer is " + random.choice([options[o] for o in options.keys() if options[o] != answer])})

        total_chosen.append(chosen)
        total_rejected.append(rejected)

    return {"chosen": total_chosen, "rejected": total_rejected}


def generate_rl_dataset_mcqa(examples, tokenizer):
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"

    rejected_total = []
    chosen_total =  []

    for question, opa, opb, opc, opd, cop, choice_type, exp, subject_name, topic_name in zip(
        examples['question'],
        examples['opa'],
        examples['opb'],
        examples['opc'],
        examples['opd'],
        examples['cop'],
        examples['choice_type'],
        examples['exp'],
        examples['subject_name'],
        examples['topic_name'],
    ):
        chosen = []
        rejected = []
        option = {}

        if question and question.strip():
            prompt = ""
            if subject_name:
                prompt = "For the subject " + subject_name + " "
            if topic_name:
                prompt += "with specific topic " + topic_name + " "
            chosen.append({"role": USER_ROLE, "content": prompt + question})
            rejected.append({"role": USER_ROLE, "content": prompt + question})
        else:
            continue

        if opa and opa != "None" and opa not in option.values():
            option["1"] = opa
        if opb and opb != "None" and opb not in option.values():
            option["2"] = opb
        if opc and opc != "None" and opc not in option.values():
            option["3"] = opc
        if opd and opd != "None" and opd not in option.values():
            option["4"] = opd

        if cop and len(option) >= 2 and str(cop) in option.keys():
            answer_c = f"The answer is {option[str(cop)]},"
            rejected_c = random.choice([option[o] for o in option.keys() if option[str(cop)] != option[o]])
        else:
            continue

        answer_e = f"and the explanation is that {exp}" if exp else ""
        answer_t = answer_c + answer_e

        chosen.append({"role": ASSISTANT_ROLE, "content": answer_t})
        rejected.append({"role": ASSISTANT_ROLE, "content": rejected_c})

        chosen_total.append(chosen)
        rejected_total.append(rejected)

    return {"chosen": chosen_total, "rejected": rejected_total}

def generate_rl_dataset_total(dataset_path_func_map: Dict[str, Callable], tokenizer, ratio):
    train_dataset = None
    test_dataset = None
    sample_size = 3000
    sample_size_usmle = math.ceil(sample_size * (ratio / (1 + ratio)))
    sample_size_mcqa = sample_size - sample_size_usmle

    for file_path, func in dataset_path_func_map.items():
        dataset = load_dataset("json", data_files=file_path, split="train", cache_dir=CACHE_DIR)
        tmp = dataset.map(
            func,
            fn_kwargs={"tokenizer": tokenizer},
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names
        )
        if "usmle" in file_path.lower():
            tmp = dataset_sample(tmp, sample_ratio=None, size=sample_size_usmle)
        elif "mcqa" in file_path.lower():
            tmp = dataset_sample(tmp, sample_ratio=None, size=sample_size_mcqa)
        splits = tmp.train_test_split(test_size=0.1, shuffle=True, seed=42)
        if train_dataset is None:
            train_dataset = splits["train"]
        else:
            train_dataset = concatenate_datasets([train_dataset, splits["train"]])
        if test_dataset is None:
            test_dataset = splits["test"]
        else:
            test_dataset = concatenate_datasets([test_dataset, splits["test"]])

    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    return train_dataset, test_dataset

def load_orpo_config():
    orpo_config = ORPOConfig(
        output_dir="../llama3-8b-medical-orpo",
        beta=0.05,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        max_length=1024,
        max_prompt_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
        generate_during_eval=True,
        num_train_epochs=1,
        bf16=True,
        report_to="tensorboard",
        gradient_checkpointing=True,
        disable_tqdm=False,
        logging_dir="../logs",
        eval_strategy="steps",
        eval_steps=20,
    )
    return orpo_config



if __name__ == "__main__":

    CACHE_DIR = r"D:\huggingface_cache"

    # combine the two datasets with ratio and implement RL mcqa 4 : 1 usmle

    dataset_func_map = {
        "../datasets/MedQA-USMLE-4-options/json/MedQA-USMLE-4-options.jsonl": generate_rl_dataset_usmle,
        "../datasets/medmcqa/json/medmcqa.jsonl": generate_rl_dataset_mcqa,
    }

    MODEL_NAME = "../gpu_med_full_model_lora"
    OUTPUT_DIR = "../llama3-8b-medical-orpo"

    tokenizer = load_tokenizer(MODEL_NAME, CACHE_DIR)
    tokenizer.padding_side = "right" # for orpo, right is better
    # for input not dialog, no need to use chat_template

    # randomly select one wrong answer as rejected message
    train_dataset, test_dataset = generate_rl_dataset_total(dataset_func_map, tokenizer=tokenizer, ratio=0.3)


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    orpo_config = load_orpo_config()
    logger.info(f"loading orpo trainer")
    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    logger.info(f"loading orpo trainer complete")
    logger.info(f"start orpo training process!")
    trainer.train()
    logger.info(f"orpo training process complete!")
    trainer.save_model(output_dir=OUTPUT_DIR)
