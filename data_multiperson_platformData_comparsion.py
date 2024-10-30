import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import track
from utils import *

def calculate_person_platform_mean(data):
    """
    计算每个人在每个平台值下的平均值
    返回格式: {platform_value: {person: mean_value}}
    """
    platform_person_means = {}
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    for platform_value in platform_values:
        platform_person_means[platform_value] = {}
        
        for person in data.keys():
            all_platform_data = []
            
            # 收集该用户在所有实验和trial中特定平台值的数据
            for exp_type in data[person]:
                for trial in data[person][exp_type]:
                    if platform_value in data[person][exp_type][trial]:
                        platform_data = data[person][exp_type][trial][platform_value]['data']
                        all_platform_data.extend(platform_data)
            
            if all_platform_data:  # 如果有数据
                platform_person_means[platform_value][person] = np.mean(all_platform_data)
    
    return platform_person_means

def calculate_person_platform_median(data):
    """
    计算每个人在每个平台值下的中值
    返回格式: {platform_value: {person: median_value}}
    """
    platform_person_medians = {}
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    for platform_value in platform_values:
        platform_person_medians[platform_value] = {}
        
        for person in data.keys():
            all_platform_data = []
            
            # 收集该用户在所有实验和trial中特定平台值的数据
            for exp_type in data[person]:
                for trial in data[person][exp_type]:
                    if platform_value in data[person][exp_type][trial]:
                        platform_data = data[person][exp_type][trial][platform_value]['data']
                        all_platform_data.extend(platform_data)
                        
            if all_platform_data:  # 如果有数据
                platform_person_medians[platform_value][person] = np.median(all_platform_data)
    
    return platform_person_medians  

def plot_platform_person_comparison(platform_person_means, save_path):
    """
    为每个平台值绘制不同用户的比较图，优化视觉效果
    """
    # 设置seaborn风格
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.figure(figsize=(16, 9))
    
    # 准备数据
    platforms = []
    persons = []
    values = []
    
    # 定义平台值的显示顺序
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    for platform in platform_order:  # 使用固定顺序
        if platform in platform_person_means:
            for person in platform_person_means[platform]:
                platforms.append(platform)
                persons.append(person)
                values.append(platform_person_means[platform][person])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Platform': platforms,
        'Person': persons,
        'Value': values
    })
    
    # 设置颜色方案
    colors = sns.color_palette("husl", n_colors=len(df['Person'].unique()))
    
    # 绘制柱状图
    ax = sns.barplot(
        data=df,
        x='Platform',
        y='Value',
        hue='Person',
        palette=colors,
        order=platform_order,
        alpha=0.8
    )
    
    # 添加标题和标签
    plt.title('Platform Values Comparison Across Persons', 
             pad=20, 
             fontsize=14, 
             fontweight='bold')
    plt.xlabel('Platform Value', fontsize=12)
    plt.ylabel('Mean Value (Offset from 0_1)', fontsize=12)
    
    # 设置x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 优化图例
    plt.legend(
        title='Participant',
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0
    )
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # 将网格线置于数据下方
    
    # 设置背景和边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, rotation=0, fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(
        os.path.join(save_path, 'platform_person_comparison_mean.png'),
        bbox_inches='tight',
        dpi=300,
        facecolor='white',
        edgecolor='none'
    )
    plt.close()
    
def plot_platform_person_comparison_median(platform_person_medians, save_path):
    """
    为每个平台值绘制不同用户的比较图，优化视觉效果
    """
    # 设置seaborn风格
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.figure(figsize=(16, 9))
    
    # 准备数据
    platforms = []
    persons = []
    values = []
    
    # 定义平台值的显示顺序
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    for platform in platform_order:  # 使用固定顺序
        if platform in platform_person_medians:
            for person in platform_person_medians[platform]:
                platforms.append(platform)
                persons.append(person)
                values.append(platform_person_medians[platform][person])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Platform': platforms,
        'Person': persons,
        'Value': values
    })
    
    # 设置颜色方案
    colors = sns.color_palette("husl", n_colors=len(df['Person'].unique()))
    
    # 绘制柱状图
    ax = sns.barplot(
        data=df,
        x='Platform',
        y='Value',
        hue='Person',
        palette=colors,
        order=platform_order,
        alpha=0.8
    )
    
    # 添加标题和标签
    plt.title('Platform Values Comparison Across Persons', 
             pad=20, 
             fontsize=14, 
             fontweight='bold')
    plt.xlabel('Platform Value', fontsize=12)
    plt.ylabel('Mean Value (Offset from 0_1)', fontsize=12)
    
    # 设置x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 优化图例
    plt.legend(
        title='Participant',
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0
    )
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # 将网格线置于数据下方
    
    # 设置背景和边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, rotation=0, fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(
        os.path.join(save_path, 'platform_person_comparison_median.png'),
        bbox_inches='tight',
        dpi=300,
        facecolor='white',
        edgecolor='none'
    )
    plt.close()

def plot_platform_person_line_comparison(platform_person_means, save_path):
    """
    为每个用户绘制平台值的折线图比较
    """
    # 设置seaborn风格
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.figure(figsize=(16, 9))
    
    # 定义平台值的显示顺序
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    # 为每个用户绘制折线
    for person in set(person for platform in platform_person_means.values() for person in platform.keys()):
        values = []
        for platform in platform_order:
            if platform in platform_person_means and person in platform_person_means[platform]:
                values.append(platform_person_means[platform][person])
            else:
                values.append(None)  # 处理缺失数据
        
        # 绘制折线
        plt.plot(range(len(platform_order)), values, marker='o', label=person, linewidth=2, markersize=8)
        
        # 添加数值标签
        for i, value in enumerate(values):
            if value is not None:
                plt.annotate(f'{value:.2f}', 
                           (i, value), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=8)
    
    # 设置图表样式
    plt.title('Platform Values Trend Across Persons', 
             pad=20, 
             fontsize=14, 
             fontweight='bold')
    plt.xlabel('Platform Value', fontsize=12)
    plt.ylabel('Mean Value (Offset from 0_1)', fontsize=12)
    
    # 设置x轴标签
    plt.xticks(range(len(platform_order)), platform_order, rotation=45, ha='right')
    
    # 优化图例
    plt.legend(
        title='Participant',
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0
    )
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(
        os.path.join(save_path, 'platform_person_comparison_line_mean.png'),
        bbox_inches='tight',
        dpi=300,
        facecolor='white',
        edgecolor='none'
    )
    plt.close()

def plot_platform_person_line_comparison_median(platform_person_medians, save_path):
    """
    为每个用户绘制平台值的折线图比较（中值版本）
    """
    # 设置seaborn风格
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.figure(figsize=(16, 9))
    
    # 定义平台值的显示顺序
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    # 为每个用户绘制折线
    for person in set(person for platform in platform_person_medians.values() for person in platform.keys()):
        values = []
        for platform in platform_order:
            if platform in platform_person_medians and person in platform_person_medians[platform]:
                values.append(platform_person_medians[platform][person])
            else:
                values.append(None)
        
        plt.plot(range(len(platform_order)), values, marker='o', label=person, linewidth=2, markersize=8)
        
        # 添加数值标签
        for i, value in enumerate(values):
            if value is not None:
                plt.annotate(f'{value:.2f}', 
                           (i, value), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=8)
    
    plt.title('Platform Values Trend Across Persons (Median)', 
             pad=20, 
             fontsize=14, 
             fontweight='bold')
    plt.xlabel('Platform Value', fontsize=12)
    plt.ylabel('Median Value (Offset from 0_1)', fontsize=12)
    
    plt.xticks(range(len(platform_order)), platform_order, rotation=45, ha='right')
    
    plt.legend(
        title='Participant',
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0
    )
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(save_path, 'platform_person_comparison_line_median.png'),
        bbox_inches='tight',
        dpi=300,
        facecolor='white',
        edgecolor='none'
    )
    plt.close()

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(base_path, "result")
    data_path = os.path.join(result_path, "statistics_result.json")
    
    fig_path = os.path.join(base_path, "fig", "multiperson_platform_comparison")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    # 加载数据
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 对数据进行偏移处理
    data = data_offset(data)
    
    # 计算每个人在每个平台值下的平均值
    platform_person_means = calculate_person_platform_mean(data)
    
    # 计算每个人在每个平台值下的中值
    platform_person_medians = calculate_person_platform_median(data)
    
    # 绘制比较图
    plot_platform_person_comparison(platform_person_means, fig_path)
    
    plot_platform_person_comparison_median(platform_person_medians, fig_path)
    
    # 添加折线图的绘制
    plot_platform_person_line_comparison(platform_person_means, fig_path)
    plot_platform_person_line_comparison_median(platform_person_medians, fig_path)

if __name__ == "__main__":
    main()


