import json
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_training_history(json_path):
    # 1. 加载数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. 提取 log_history 并转为 DataFrame
    df = pd.DataFrame(data['log_history'])

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))

    # --- 图 1: Loss 趋势 ---
    if 'loss' in df.columns:
        train_loss = df[df['loss'].notna()]
        # df['loss_smooth'] = df['loss'].rolling(window=10).mean()
        # train_smooth_loss = df[df['loss_smooth'].notna()]
        axes[0][0].plot(train_loss['step'], train_loss['loss'], label='Training Loss', color='#1f77b4', alpha=0.8)

    if 'eval_loss' in df.columns:
        eval_loss = df[df['eval_loss'].notna()]
        axes[0][0].plot(eval_loss['step'], eval_loss['eval_loss'], label='Eval Loss', marker='o', color='#ff7f0e')

    axes[0][0].set_title('Training & Evaluation Loss')
    axes[0][0].set_xlabel('Steps')
    axes[0][0].set_ylabel('Loss')
    axes[0][0].legend()

    # --- 图 2: 学习率 (Learning Rate) 趋势 ---
    if 'learning_rate' in df.columns:
        lr_df = df[df['learning_rate'].notna()]
        axes[0][1].plot(lr_df['step'], lr_df['learning_rate'], color='#2ca02c')
        axes[0][1].set_title('Learning Rate Schedule')
        axes[0][1].set_xlabel('Steps')
        axes[0][1].set_ylabel('LR')
        # 使用科学计数法
        axes[0][1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


    if 'grad_norm' in df.columns:
        lr_df = df[df['learning_rate'].notna()]
        axes[1][0].plot(lr_df['step'], lr_df['grad_norm'], color='#2ca02c')
        axes[1][0].set_title('Grad Norm Rate Schedule')
        axes[1][0].set_xlabel('Steps')
        axes[1][0].set_ylabel('LR')
        # 使用科学计数法
        axes[1][0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if 'mean_token_accuracy' in df.columns:
        lr_df = df[df['learning_rate'].notna()]
        axes[1][1].plot(lr_df['step'], lr_df['mean_token_accuracy'], color='#2ca02c')
        axes[1][1].set_title('Mean Token Accuracy Rate Schedule')
        axes[1][1].set_xlabel('Steps')
        axes[1][1].set_ylabel('LR')
        # 使用科学计数法
        axes[1][1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()

# 使用方法：将路径替换为你自己的 trainer_state.json 路径
if __name__ == '__main__':
    plot_training_history('../gpu_med_lora_results_final/checkpoint-674/trainer_state.json')