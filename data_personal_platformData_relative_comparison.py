import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import track
from utils import *

def calculate_platform_steps(data, user_name):
    """
    计算平台之间的步进值
    正向序列: 0_1->5->10->15->25
    负向序列: 0_2->-5->-10->-15->-25
    """
    positive_sequence = ["0_1", "5", "10", "15", "25"]
    negative_sequence = ["0_2", "-5", "-10", "-15", "-25"]
    
    steps_data = {
        'positive': {},  # {step_name: [各次实验的值]}
        'negative': {}   # {step_name: [各次实验的值]}
    }
    
    # 初始化步进名称
    for i in range(len(positive_sequence)-1):
        steps_data['positive'][f"{positive_sequence[i]}->{positive_sequence[i+1]}"] = []
        steps_data['negative'][f"{negative_sequence[i]}->{negative_sequence[i+1]}"] = []
    
    user_data = data[user_name]
    for exp_type in user_data:
        for trial in user_data[exp_type]:
            # 计算正向序列的步进值
            for i in range(len(positive_sequence)-1):
                platform1 = positive_sequence[i]
                platform2 = positive_sequence[i+1]
                if (platform1 in user_data[exp_type][trial] and 
                    platform2 in user_data[exp_type][trial]):
                    mean1 = np.mean(user_data[exp_type][trial][platform1]['data'])
                    mean2 = np.mean(user_data[exp_type][trial][platform2]['data'])
                    step_name = f"{platform1}->{platform2}"
                    steps_data['positive'][step_name].append(mean2 - mean1)
            
            # 计算负向序列的步进值
            for i in range(len(negative_sequence)-1):
                platform1 = negative_sequence[i]
                platform2 = negative_sequence[i+1]
                if (platform1 in user_data[exp_type][trial] and 
                    platform2 in user_data[exp_type][trial]):
                    mean1 = np.mean(user_data[exp_type][trial][platform1]['data'])
                    mean2 = np.mean(user_data[exp_type][trial][platform2]['data'])
                    step_name = f"{platform1}->{platform2}"
                    steps_data['negative'][step_name].append(mean2 - mean1)
    
    return steps_data

def plot_step_comparison_bars_with_cumulative(steps_data, user_name, save_path):
    """
    使用柱状图对比不同实验的步进值，并添加累积值的折线图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f'Platform Step Differences and Cumulative Values for {user_name}', fontsize=16)
    
    # 设置柱状图的宽度和颜色
    bar_width = 0.8 / max(
        max(len(values) for values in steps_data['positive'].values()),
        max(len(values) for values in steps_data['negative'].values())
    )
    colors = plt.cm.Set3(np.linspace(0, 1, max(
        max(len(values) for values in steps_data['positive'].values()),
        max(len(values) for values in steps_data['negative'].values())
    )))
    
    # 创建双Y轴
    ax1_twin = ax1.twinx()
    ax2_twin = ax2.twinx()
    
    # 绘制正向序列
    for step_idx, (step_name, values) in enumerate(steps_data['positive'].items()):
        # 绘制柱状图
        for exp_idx, value in enumerate(values):
            x = step_idx + exp_idx * bar_width
            bar = ax1.bar(x, value, bar_width, color=colors[exp_idx], 
                         label=f'Trial {exp_idx+1}' if step_idx == 0 else "")
            ax1.text(x, value, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 计算并绘制累积值
        for exp_idx in range(len(values)):
            cumulative_values = np.cumsum([steps_data['positive'][key][exp_idx] 
                                         for key in steps_data['positive'].keys()])
            x_points = np.arange(len(cumulative_values)) + exp_idx * bar_width
            ax1_twin.plot(x_points, cumulative_values, 'o-', color=colors[exp_idx],
                         label=f'Cumulative Trial {exp_idx+1}' if step_idx == len(steps_data['positive'])-1 else "")
            # 添加累积值标签
            for x, y in zip(x_points, cumulative_values):
                ax1_twin.text(x, y, f'{y:.1f}', ha='right', va='bottom', fontsize=8)
    
    # 设置正向序列图表属性
    ax1.set_title('Positive Sequence Steps')
    ax1.set_xticks([i + (len(values) - 1) * bar_width / 2 for i, values in enumerate(steps_data['positive'].values())])
    ax1.set_xticklabels(steps_data['positive'].keys())
    ax1.set_ylabel('Step Difference')
    ax1_twin.set_ylabel('Cumulative Value')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    # ax1_twin.legend(loc='upper right')
    
    # 绘制负向序列
    for step_idx, (step_name, values) in enumerate(steps_data['negative'].items()):
        # 绘制柱状图
        for exp_idx, value in enumerate(values):
            x = step_idx + exp_idx * bar_width
            bar = ax2.bar(x, value, bar_width, color=colors[exp_idx],
                         label=f'Trial {exp_idx+1}' if step_idx == 0 else "")
            ax2.text(x, value, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 计算并绘制累积值
        for exp_idx in range(len(values)):
            cumulative_values = np.cumsum([steps_data['negative'][key][exp_idx] 
                                         for key in steps_data['negative'].keys()])
            x_points = np.arange(len(cumulative_values)) + exp_idx * bar_width
            ax2_twin.plot(x_points, cumulative_values, 'o-', color=colors[exp_idx],
                         label=f'Cumulative Trial {exp_idx+1}' if step_idx == len(steps_data['negative'])-1 else "")
            # 添加累积值标签
            for x, y in zip(x_points, cumulative_values):
                ax2_twin.text(x, y, f'{y:.1f}', ha='right', va='bottom', fontsize=8)
    
    # 设置负向序列图表属性
    ax2.set_title('Negative Sequence Steps')
    ax2.set_xticks([i + (len(values) - 1) * bar_width / 2 for i, values in enumerate(steps_data['negative'].values())])
    ax2.set_xticklabels(steps_data['negative'].keys())
    ax2.set_ylabel('Step Difference')
    ax2_twin.set_ylabel('Cumulative Value')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')
    # ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'platform_steps_comparison_with_cumulative.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    base_path = os.path.dirname(__file__)
    result_path = os.path.join(base_path, "result")
    data_path = os.path.join(result_path, "statistics_result.json")
    
    fig_path = os.path.join(base_path, "fig", "personal_platform_relative_comparison")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    # 加载数据
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 对数据进行偏移处理
    data = data_offset(data)
    
    # 为每个用户生成步进值分析
    for user_name in track(data.keys(), description="Analyzing platform steps for each user"):
        # 创建用户专属文件夹
        user_fig_path = os.path.join(fig_path, user_name)
        if not os.path.exists(user_fig_path):
            os.makedirs(user_fig_path)
        
        # 计算步进值
        steps_data = calculate_platform_steps(data, user_name)
        
        # 生成柱状图对比
        plot_step_comparison_bars_with_cumulative(steps_data, user_name, user_fig_path)

if __name__ == "__main__":
    main()