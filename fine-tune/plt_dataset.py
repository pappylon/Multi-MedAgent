import math
from typing import final

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger



def formatting_prompts_func_medicalqa(examples, tokenizer):
    '''
    23w的medical-qa数据集的prompt处理
    :param examples:
    :param tokenizer:
    :return:
    '''
    texts = []
    token_len = []

    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    CUT_SIZE = 100000
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

        formatted_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False, # 如若还需以新的prompt进行设置为true
        )

        if len(formatted_text) <= CUT_SIZE:
            texts.append(formatted_text)
            token_len.append(len(formatted_text))

    return {"messages": texts, "token_length": token_len}

def formatting_prompts_func_medquad(examples, tokenizer):
    '''
    1w6的medquad数据集处理
    :param examples:
    :param tokenizer:
    :return:
    '''
    texts = []
    token_len = []

    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    CUT_SIZE = 100000

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
            # return_assistant_tokens_mask=True,
            # return_dict=True,
            # return_tensors="pt",
        )
        if len(formatted_text) <= CUT_SIZE:
            texts.append(message)
            token_len.append(len(formatted_text))

        # # 标准的Instruction Tuning Prompt格式
        # if input_data and input_data.strip():
        #     # 有输入时的 Prompt 模板
        #     prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_data}\n### Output:\n{output}\n"
        # else:
        #     # 无输入时的 Prompt 模板
        #     prompt = f"### Instruction:\n{instruction}\n### Output:\n{output}\n"
        # texts.append(prompt)

    return {"messages": texts, "token_length": token_len}

def formatting_prompts_func_medicalo1(examples, tokenizer):
    texts = []
    token_len = []

    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    CUT_SIZE = 100000

    # if not all(k in examples and examples[k] is not None for k in ["Question", "Complex_CoT", "Response"]):
    #     logger.debug(f"数据集 medical-o1 部分内容有缺失")
    for question, cot, answer in zip(examples["Question"], examples["Complex_CoT"], examples["Response"]):
        assistant_content = f"<thought>\n{cot}\n</thought>\n\n{answer}"
        text = (
            f"<|begin_of_text|><|start_header_id|>{SYSTEM_ROLE}<|end_header_id|>You are a professional medical expert. Prior to answering the question, please first present a detailed clinical reasoning process (Chain of Thought), and then provide the final answer。<|eot_id|>"
            f"<|start_header_id|>{USER_ROLE}<|end_header_id|>\n\n{question}<|eot_id|>"
            f"<|start_header_id|>{ASSISTANT_ROLE}<|end_header_id|>\n\n{assistant_content}<|eot_id|>"
        )

        if len(text) <= CUT_SIZE:
            texts.append(text)
            token_len.append(len(text))
        # # 标准的Instruction Tuning Prompt格式
        # if input_data and input_data.strip():
        #     # 有输入时的 Prompt 模板
        #     prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_data}\n### Output:\n{output}\n"
        # else:
        #     # 无输入时的 Prompt 模板
        #     prompt = f"### Instruction:\n{instruction}\n### Output:\n{output}\n"
        # texts.append(prompt)

    return {"messages": texts, "token_length": token_len}

def formatting_prompts_func_pubmedqa(examples, tokenizer):
    texts = []
    token_len = []
    CUT_SIZE = 100000
    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"

    # if not all(k in examples and examples[k] is not None for k in ["question", "context", "long_answer", "final_decision"]):
    #     logger.debug(f"数据集 pubmedqa 部分内容有缺失")

    for question, context, answer, decision in zip(examples["question"], examples["context"], examples["long_answer"], examples["final_decision"]):
        structured_bg = ""
        for label, text in zip(context['labels'], context['contexts']):
            # 加上清晰的标题分隔符，帮助模型识别逻辑段落
            structured_bg += f"### {label}:\n{text}\n\n"

        keywords = ", ".join(context['meshes'])
        sys_prompt = "You are a clinical medicine expert. Please answer the question based on the provided research abstract (Background) and keywords. The analysis must be accurate and logically rigorous, and a final interpretation must be given."

        user_content = (
            f"【research background】:\n{structured_bg}"
            f"【keywords】: {keywords}\n\n"
            f"【question】: {question}\n\n"
            f"Please conduct a detailed analysis of the study's results based on the aforementioned evidence and provide a final conclusion."
        )

        # 5. 拼接 Llama-3 官方格式
        # 假设 example['long_answer'] 和 example['final_decision'] 是标签
        assistant_content = (
            f"【analysis】: {answer}\n\n"
            f"final result：[{decision}]"
        )

        full_text = (
            f"<|begin_of_text|><|start_header_id|>{SYSTEM_ROLE}<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
            f"<|start_header_id|>{USER_ROLE}<|end_header_id|>\n\n{user_content}<|eot_id|>"
            f"<|start_header_id|>{ASSISTANT_ROLE}<|end_header_id|>\n\n{assistant_content}<|eot_id|>"
        )
        if len(full_text) <= CUT_SIZE:
            texts.append(full_text)
            token_len.append(len(full_text))

    return {"messages": texts, "token_length": token_len}

def plot_dataset_feature(title, lengths) -> None:
    p50 = np.percentile(lengths, 50)
    p70 = np.percentile(lengths, 70)
    p80 = np.percentile(lengths, 80)
    p90 = np.percentile(lengths, 90)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)
    max_len = np.max(lengths)
    min_len = np.min(lengths)

    print(f"--- 数据集 ---{title}--- Token 长度统计 ---")
    print(f"最小长度: {min_len}")
    print(f"中位数 (P50): {p50}")
    print(f"90% 分位数 (P90): {p90}")
    print(f"95% 分位数 (P95): {p95}")
    print(f"99% 分位数 (P99): {p99}")
    print(f"最大长度: {max_len}")

    # 5. 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(p95, color='red', linestyle='dashed', linewidth=1, label=f'95th Percentile ({int(p95)})')
    plt.axvline(p80, color='red', linestyle='dashed', linewidth=1, label=f'80th Percentile ({int(p80)})')
    plt.axvline(p70, color='red', linestyle='dashed', linewidth=1, label=f'80th Percentile ({int(p70)})')
    plt.axvline(p50, color='red', linestyle='dashed', linewidth=1, label=f'50th Percentile ({int(p50)})')

    plt.title(f"dataset {title} token Length Distribution")
    plt.xlabel("Length (tokens)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)


#对数据进行采样
def dataset_sample(dataset, sample_ratio, sample_size: int = 2000) -> Dataset:
    if sample_ratio != 0:
        sample_size = math.ceil(len(dataset) * sample_ratio)
    all_indices = list(range(len(dataset)))
    sampled_indices = random.sample(all_indices, sample_size)
    sampled_dataset = dataset.select(sampled_indices)
    logger.info(f"采样后的数据集长度为: {len(sampled_dataset)}")
    return sampled_dataset

def plt_dataset_ratio(data_ratio_map) -> None:
    # 1. 准备数据
    labels = []
    sizes = []
    for key, value in data_ratio_map.items():
        labels.append(key)
        sizes.append(value)

    # 2. 设置绘图区域
    plt.figure(figsize=(8, 8))  # 设置画布大小，让饼图看起来更圆

    # 3. 绘制饼图
    # autopct='%1.1f%%'：设置百分比格式，保留一位小数
    # startangle=90：设置起始角度，从顶部开始画
    # colors：为每个扇区指定颜色
    # explode：突出显示某个扇区 (可选)

    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'],  # 自定义颜色
        wedgeprops={'edgecolor': 'black'}  # 设置扇区边缘线
    )

    # 4. 设置图表标题
    plt.title('Dataset ratio', fontsize=16)

    # 5. 确保饼图是正圆 (这非常关键)
    plt.axis('equal')

    # 6. 显示图例 (可选)
    plt.legend(loc='upper right', title="Ratio")

if __name__ == "__main__":
    CACHE_DIR = r"D:\huggingface_cache"
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

    dataset_path_map = {
        '../datasets/MedQuad-MedicalQnADataset/json/MedQuad-MedicalQnADataset.jsonl':formatting_prompts_func_medquad,
        '../datasets/medical-qa-datasets-all-processed/json/medical-qa-datasets.jsonl':formatting_prompts_func_medicalqa,
        '../datasets/medical-o1-reasoning-SFT-en/json/medical-o1-reasoning-SFT.jsonl':formatting_prompts_func_medicalo1,
        '../datasets/PubMedQA-pqa_artificial/json/PubMedQA.jsonl':formatting_prompts_func_pubmedqa,
    }
    lora_weights_path = "../gpu_med_lora_results"  # 上一步保存 LoRA 权重的本地路径

    merge_full_model_save_dir = "../gpu_med_full_model"

    # 2. 加载基础模型和 Tokenizer
    # 注意：加载基础模型时，如果训练时使用了 4bit 量化（QLoRA），推理时最好也使用相同的量化配置，以确保内存和兼容性。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = (
    #         "{% if not add_generation_prompt %}<|begin_of_text|>{% endif %}"
    #         "{% for message in messages %}"
    #         "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    #         "{% endfor %}"
    #         "{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}"
    #         "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    #         "{% endif %}"
    #     )


    # print(len(dataset))
    # print(dataset[0])
    # print(formatting_prompts_func(dataset[0]))
    import random
    SAMPLE_SIZE = 1000

    total_data = []
    data_title_ratio_map = {}

    train_totle = None
    test_totle = None

    for file_path, func in dataset_path_map.items():
        title = file_path.split("/")[-1].split(".")[0]
        print(title)
        dataset = load_dataset("json", data_files=file_path, split="train", cache_dir=CACHE_DIR)
        dataset = dataset.shuffle(seed=42)
        print(dataset.column_names)


        tmp = dataset.map(
            func,
            batched=True,
            batch_size=2000,
            num_proc=4,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=dataset.column_names,
        )
        print(len(tmp))
        print(tmp['messages'][0])
        print(tmp['messages'].column_name)


        split = tmp.train_test_split(test_size=0.1, shuffle=True, seed=42)
        # if train_totle is None:
        #     train_totle = split["train"]
        # else:
        #     train_totle = concatenate_datasets([train_totle, split["train"]])
        # if test_totle is None:
        #     test_totle = split["test"]
        # else:
        #     test_totle = concatenate_datasets([test_totle, split["test"]])
        # print(split['train'])
        # print(split['test'])
        #
        # print(train_totle)
        # print(train_totle['messages'][0])
        # print(test_totle)


        data_title_ratio_map[title] = len(tmp['messages'])

        total_data.extend(tmp['messages'])
        print(tmp)
        print(tmp['messages'][0])
        print((len(tmp['messages'])))
        lengths = tmp['token_length']
        plot_dataset_feature(title, lengths)

    plt_dataset_ratio(data_title_ratio_map)
    print(len(total_data))
    plt.show()




