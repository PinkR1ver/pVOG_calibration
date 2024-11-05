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

    for exp_type in user_data.keys():
        for trial in user_data[exp_type]:
            
            if user_name == 'wdy' and exp_type == '1':
                continue
            
            if platform_value in user_data[exp_type][trial]:
                platform_data = user_data[exp_type][trial][platform_value]['data']
                plot_data.append(platform_data)
    
    if not plot_data:  # 如果没有数据，直接返回
        return
        
    plt.figure(figsize=(10, 6))
    
    if user_name != 'wdy':
    
        # 使用原始数据绘制箱型图
        box_plot = plt.boxplot(
            plot_data,
            labels=[f'Exp{exp_type}-{trial}' 
                    for exp_type in user_data.keys()
                    for trial in user_data[exp_type] 
                    if platform_value in user_data[exp_type][trial]],
            patch_artist=True
        )
        
    else:
        exp_type = '2'
        box_plot = plt.boxplot(
            plot_data,
            labels=[f'Exp{exp_type}-{trial}' 
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
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f'platform_{platform_value}.png'),
                bbox_inches='tight')
    plt.close()

def plot_all_platforms_subplots(data, user_name, save_path):
    """
    为指定用户创建单张大图，包含所有平台值的箱型图
    x轴为目标角度，y轴为实际角度
    """
    platform_values = ["5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    plt.figure(figsize=(7, 4))
    
    user_data = data[user_name]
    exp_type = '1'
    
    # 准备数据
    all_positions = []
    all_data = []
    all_labels = []
    
    # 获取所有可能的trial编号
    all_trials = sorted(list(user_data[exp_type].keys()))
    
    # 将platform_values转换为数值并排序
    platform_nums = []
    for pv in platform_values:
        if '_' in pv:
            platform_nums.append(0)
        else:
            platform_nums.append(float(pv))
    
    sorted_indices = np.argsort(platform_nums)
    platform_values = [platform_values[i] for i in sorted_indices]
    platform_nums = [platform_nums[i] for i in sorted_indices]
    
    # 为每个平台值的每次实验准备数据
    for platform_value, target_angle in zip(platform_values, platform_nums):
        for trial in all_trials:
            if platform_value in user_data[exp_type][trial]:
                all_data.append(user_data[exp_type][trial][platform_value]['data'])
                all_positions.append(target_angle)
                all_labels.append(f'Trial {trial}')
    
    if all_data:
        # 使用原始数据绘制箱型图
        box_plot = plt.boxplot(
            all_data,
            positions=all_positions,
            patch_artist=True,
            widths=0.8
        )
        
        # 设置不同trial的颜色
        unique_trials = sorted(list(set(all_labels)))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_trials)))
        trial_colors = {trial: color for trial, color in zip(unique_trials, colors)}
        
        # 为每个箱型图设置颜色
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            if element in box_plot:
                if element == 'boxes':
                    for box, label in zip(box_plot[element], all_labels):
                        box.set_facecolor(trial_colors[label])
                        box.set_alpha(0.7)
                else:
                    for item, label in zip(box_plot[element], all_labels):
                        item.set_color(trial_colors[label])
                        if element == 'medians':
                            item.set_color('black')  # 保持中位数线为黑色以便清晰显示
        
        # 添加理想值的对角线
        min_val = min(min(platform_nums), min([min(d) for d in all_data]))
        max_val = max(max(platform_nums), max([max(d) for d in all_data]))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal')
        
        plt.title(f'Platform Comparison for {user_name}')
        plt.xlabel('Target Angle (degrees)')
        plt.ylabel('Measured Angle (degrees)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 创建图例
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=label)
                         for label, color in trial_colors.items()]
        legend_elements.append(plt.Line2D([0], [0], linestyle='--', color='k', label='Ideal'))
        plt.legend(handles=legend_elements, loc='best')
        
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
        # plot_all_platforms_subplots(data, user_name, user_fig_path) 
    
    
    
    
