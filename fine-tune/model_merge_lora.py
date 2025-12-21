import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel



def save_merge_model(base_model_path, lora_adaptor_path, output_dir, cache_dir):
    logger.info(f"start loading tokenizer from lora path")
    tokenizer = AutoTokenizer.from_pretrained(lora_adaptor_path)
    logger.info(f"loading tokenizer success!")

    logger.info(f"start loading base model!")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        dtype=torch.bfloat16,  # dtype
        cache_dir=cache_dir,
        # offload_folder="offload_folder",
    )
    logger.info(f"loading base model success!")

    logger.info("start loading lora adaptor")
    model = PeftModel.from_pretrained(
        base_model,
        lora_adaptor_path,
        dtype=torch.bfloat16,
    )
    logger.info("lora adaptor load success!")


    merged_model = model.merge_and_unload()
    merged_model.trainable_params = 0 # 可选：确保模型被视为全参数模型
    logger.info(f"QLoRA model merges successfully!")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"full accuracy model saved successfully!")


if __name__ == "__main__":
    MODLE_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # 原始的基础模型 ID
    lora_weights_path = "../gpu_med_lora_results_final"  # 上一步保存 LoRA 权重的本地路径
    output_dir = "../gpu_med_full_model_lora"
    CACHE_DIR = r"D:\huggingface_cache"
    save_merge_model(MODLE_NAME, lora_adaptor_path=lora_weights_path, output_dir=output_dir, cache_dir=CACHE_DIR)
