from typing import Dict, List

import torch
import time

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from evaluate import load
from tqdm import tqdm

from bert_score import score
from loguru import logger
# from moverscore_v2 import get_idf_dict, word_mover_score

from model import (
    dataset_sample,
    load_total_dataset,
    dataset_path_func_map,
    load_tokenizer,
)


# --- 配置参数 ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = r"D:\huggingface_cache"
OUTPUT_DIR = "../gpu_med_lora_results_final"
MODEL_ID = "../gpu_med_full_model_lora"


MAX_NEW_TOKENS = 256
TEMPERATURE = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- 评测配置 ---")
print(f"模型: {MODEL_ID}")
print(f"设备: {DEVICE}")
print(f"最大生成 Token 数: {MAX_NEW_TOKENS}\n")

## --- 1. 加载模型和分词器 ---
# ⚠️ 注意：如果您在 GPU 显存不足 (如低于 40GB) 的情况下，
# 可以取消注释以下代码块，使用 4-bit 量化加载模型以节省资源。

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("loading base model and tokenizer...")
try:
    tokenizer = load_tokenizer(MODEL_ID, CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        dtype=torch.bfloat16, # with cuda,
        device_map="auto",
        cache_dir=CACHE_DIR,
    ).to(DEVICE)

    # 定义 Llama 3 的终止符
    TERMINATORS = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
except Exception as e:
    logger.error(f"model load fail: {e}")
    exit()

# load
logger.info(f"Loading lora param for model")
# lora_weights_path = "../gpu_med_lora_results_final"
lora_weights_path_orpo = "../llama3-8b-medical-orpo"
model = PeftModel.from_pretrained(
        model,
        lora_weights_path_orpo,
    )
model.eval()
logger.info(f"Loading model success")

def get_dataset_answer(dataset):
    tmp = []
    for message in dataset['messages']:
        for item in message:
            if item['role'] == 'assistant':
                tmp.append(item['content'])

    return tmp

def remove_assistant_tag(example):
    total = []
    for message in example['messages']:
        mes = [item for item in message if item['role'] != 'assistant']
        total.append(mes)
    return {"messages": total}

# use train_dataset and test_dataset separately
train_dataset, test_dataset = load_total_dataset(dataset_path_func_map, tokenizer)
input_messages_train = dataset_sample(train_dataset, sample_ratio=None ,size=5)
# input_messages_test = dataset_sample(test_dataset, sample_ratio=None ,size=10000)
dataset_answer = get_dataset_answer(input_messages_train)
input_messages = input_messages_train.map(remove_assistant_tag,batched=True)
input_messages = input_messages['messages'][:]

# 从traindataset取2000条出来，再从test_dataset取2000条分别进行评估

if tokenizer.pad_token is None:
    # 开放式生成通常将 eos_token 作为 pad_token 使用
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ 成功设置 tokenizer.pad_token 为 eos_token。")

# 同时更新模型的配置，确保 model.generate 内部也知道 pad_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id
    print("✅ 成功设置 model.config.pad_token_id。")

# 应用聊天模板并获取输入 ID
tokenizer.padding_side = "left"
inputs = tokenizer.apply_chat_template(
    input_messages,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=4096,
    return_dict=True,
).to(DEVICE)

# print(tokenizer.decode(inputs[0], skip_special_tokens=False))
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# attention_mask = torch.ones_like(input_ids).to(model.device)

## --- 3. 性能评测 ---
# 确保 GPU 预热，以获得更准确的计时
if DEVICE == "cuda":
    _ = model.generate(input_ids[0:1, :], max_new_tokens=MAX_NEW_TOKENS)
    torch.cuda.synchronize()

print("\n--- 开始评测 ---")
start_time = time.time()


total_num = input_ids.size(0)
sample_size = 1
all_outputs = []
# 通过tqdm显式看到处理进度
for i in tqdm(range(0, total_num, sample_size), desc='generating output in batches'):
    batch_input_ids = input_ids[i:i + sample_size]
    batch_attension_mask = attention_mask[i:i + sample_size]
    outputs = model.generate(
        batch_input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=TERMINATORS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=0.9,
        top_k=50,
        output_scores=True,
        return_dict_in_generate=True,
        attention_mask=batch_attension_mask,
    )
    all_outputs.extend(outputs.sequences.cpu())

# outputs = model.generate(
#     input_ids,
#     max_new_tokens=MAX_NEW_TOKENS,
#     eos_token_id=TERMINATORS,
#     do_sample=True,
#     temperature=TEMPERATURE,
#     top_p=0.9,
#     top_k=50,
#     output_scores=True,
#     return_dict_in_generate=True,
#     attention_mask=attention_mask,
# )

# 计时结束
if DEVICE == "cuda":
    torch.cuda.synchronize()
end_time = time.time()

# calculate the bleu score between generated answer and standard answer
def compute_scores(predictions: List[str], references :List[str]):
    results : Dict[str, float] = {}

    rouge_metrics = load("rouge")
    rouge_scores = rouge_metrics.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    results['ROUGE-1 F1'] = rouge_scores['rouge1']
    results['ROUGE-2 F1'] = rouge_scores['rouge2']
    results['ROUGE-L F1'] = rouge_scores['rougeL']
    bleu_metric = load("sacrebleu")
    bleu_scores = bleu_metric.compute(
        predictions=predictions,
        references=references
    )
    results['bleu_score'] = bleu_scores['score'] / 100.0

    # bertscore
    P, R, F1 = score(predictions, references, lang="en", verbose=True, rescale_with_baseline=True)
    results['precision'] = P.mean()
    results['recall'] = R.mean()
    results['f1-score'] = F1.mean()

    # moverscore
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # idf_dict_ref = get_idf_dict(references)
    # predictions_List:List[List[str]] = [predictions]
    # idf_dict_hyp = get_idf_dict(predictions_List)
    #
    # model_name = 'bert-large-uncased'
    # scores = word_mover_score(
    #     references,
    #     predictions,
    #     idf_dict_ref,
    #     idf_dict_hyp,
    #     stop_words=[],  # 停用词列表，留空表示不移除
    #     n_gram=1,  # 使用 1-gram
    #     remove_subwords=True,
    #     model_type=model_name,
    #     device=device
    # )
    # results['moverscore'] = np.mean(scores)

    return results




## --- 4. 结果展示 ---
# response_tokens = outputs[0][0][input_ids.shape[-1]:]
# generated_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
generated_text_list = []
token_len = 0
for i in range(len(all_outputs)):
    response = all_outputs[i]
    token_len += len(response)
    generated_text = tokenizer.decode(response[input_ids.shape[-1]:], skip_special_tokens=True)
    generated_text_list.append(generated_text.strip())

    print(f"model prompt: {input_messages[i]}")

    print(f"standard answer: {dataset_answer[i]}\n")
    print(f"dialog number {i}: {generated_text.strip()}\n\n")


# print out the score
results = compute_scores(generated_text_list, dataset_answer)
print("\n--- Generation Model Evaluation Results ---")
for metric, score in results.items():
    print(f"| {metric.ljust(15)} | {score:.4f} |")
print("-----------------------------------------")

# print(generated_text_list)
# 计算指标
total_time = end_time - start_time
tokens_per_second = token_len / total_time


print("\n--- 评测结果 ---")
print(f"**总生成时间**: {total_time:.4f} 秒")
print(f"**生成 Token 数**: {token_len}")
print(f"**生成速度**: {tokens_per_second:.2f} tokens/秒")

print("\n--- 模型输出 ---")