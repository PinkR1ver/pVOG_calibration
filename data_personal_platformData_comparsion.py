import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import track
from utils import *

def plot_platform_comparison(data, user_name, platform_value, save_path):
    """
    为指定用户的特定平台值绘制箱型图对比
    使用原始数据绘制
    """
    # 准备数据
    plot_data = []
    user_data = data[user_name]
    
    # 遍历所有实验和trial
    for exp_type in user_data:
        for trial in user_data[exp_type]:
            if platform_value in user_data[exp_type][trial]:
                platform_data = user_data[exp_type][trial][platform_value]['data']
                plot_data.append(platform_data)
    
    if not plot_data:  # 如果没有数据，直接返回
        return
        
    plt.figure(figsize=(10, 6))
    
    # 使用原始数据绘制箱型图
    box_plot = plt.boxplot(
        plot_data,
        labels=[f'Exp{exp_type}-{trial}' 
                for exp_type in user_data 
                for trial in user_data[exp_type] 
                if platform_value in user_data[exp_type][trial]],
        patch_artist=True
    )
    
    # 设置颜色
    color = plt.cm.Set3(np.random.rand())
    for box in box_plot['boxes']:
        box.set_facecolor(color)
    
    plt.title(f'Platform {platform_value} Comparison for {user_name}')
    plt.xlabel('Experiment-Trial')
    plt.ylabel('Value (Offset from 0_1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f'platform_{platform_value}.png'),
                bbox_inches='tight')
    plt.close()

def plot_all_platforms_subplots(data, user_name, save_path):
    """
    为指定用户创建包含所有平台值的子图
    使用原始数据绘制
    """
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    fig.suptitle(f'All Platform Comparisons for {user_name}', fontsize=16)
    
    axes_flat = axes.flatten()
    user_data = data[user_name]
    
    for idx, platform_value in enumerate(platform_values):
        ax = axes_flat[idx]
        
        # 准备数据
        plot_data = []
        labels = []
        
        for exp_type in user_data:
            for trial in user_data[exp_type]:
                if platform_value in user_data[exp_type][trial]:
                    plot_data.append(user_data[exp_type][trial][platform_value]['data'])
                    labels.append(f'Exp{exp_type}-{trial}')
        
        if plot_data:
            # 使用原始数据绘制箱型图
            box_plot = ax.boxplot(
                plot_data,
                labels=labels,
                patch_artist=True
            )
            
            # 设置颜色
            color = plt.cm.Set3(np.random.rand())
            for box in box_plot['boxes']:
                box.set_facecolor(color)
            
            ax.set_title(f'Platform {platform_value}')
            ax.set_xlabel('Experiment-Trial')
            ax.set_ylabel('Value (Offset from 0_1)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f'all_platforms_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)
    result_path = os.path.join(base_path, "result")
    data_path = os.path.join(result_path, "statistics_result.json")
    
    fig_path = os.path.join(base_path, "fig")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        
    fig_path = os.path.join(fig_path, "personal_platform_comparison")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 对数据进行偏移处理
    data = data_offset(data)
    
    # 定义所有可能的平台值
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    # 为每个用户创建单独的文件夹并生成图表
    for user_name in track(data.keys(), description="Generating box plot of each platform for each user", total=len(data.keys())):
        # 创建用户专属文件夹
        user_fig_path = os.path.join(fig_path, user_name)
        if not os.path.exists(user_fig_path):
            os.makedirs(user_fig_path)
            
        # 生成单独的平台值图表
        for platform_value in platform_values:
            plot_platform_comparison(data, user_name, platform_value, user_fig_path)
            
        # 生成包含所有平台值的大图
        plot_all_platforms_subplots(data, user_name, user_fig_path)
    
    
    
    
