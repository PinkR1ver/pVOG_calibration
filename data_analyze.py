import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_platform_comparison(data, user_name, platform_value, save_path):
    """
    为指定用户的特定平台值绘制箱型图对比
    data: statistics_result.json的数据
    user_name: 用户名
    platform_value: 平台值
    save_path: 保存路径
    """
    # 准备数据
    plot_data = []
    user_data = data[user_name]
    
    # 遍历所有实验和trial
    for exp_type in user_data:
        for trial in user_data[exp_type]:
            if platform_value in user_data[exp_type][trial]:
                platform_data = user_data[exp_type][trial][platform_value]
                plot_data.append({
                    'Trial': f'Exp{exp_type}-{trial}',
                    'Median': platform_data['median'],
                    'Q1': platform_data['q1'],
                    'Q3': platform_data['q3'],
                    'Min': platform_data['min'],
                    'Max': platform_data['max']
                })
    
    if not plot_data:  # 如果没有数据，直接返回
        return
        
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制箱型图
    box_plot = plt.boxplot(
        [[row['Min'], row['Q1'], row['Median'], row['Q3'], row['Max']]
         for _, row in df.iterrows()],
        labels=df['Trial'],
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
    
    # 保存图片
    plt.savefig(os.path.join(save_path, f'platform_{platform_value}.png'),
                bbox_inches='tight')
    plt.close()

def data_offset(data):
    '''
    将数据整体偏移到0_1为0
    '''
    offset_data = json.loads(json.dumps(data))  # 深拷贝数据
    
    for user_name in offset_data:
        for exp_type in offset_data[user_name]:
            for trial_num in offset_data[user_name][exp_type]:
                offset = offset_data[user_name][exp_type][trial_num]['0_1']['mean']
                for platform_value in offset_data[user_name][exp_type][trial_num]:
                    for key in ['median', 'mean', 'q1', 'q3', 'min', 'max']:
                        offset_data[user_name][exp_type][trial_num][platform_value][key] -= offset
                        
                    offset_data[user_name][exp_type][trial_num][platform_value]['data'] = [x - offset for x in offset_data[user_name][exp_type][trial_num][platform_value]['data']]
                    
    return offset_data

def plot_all_platforms_subplots(data, user_name, save_path):
    """
    为指定用户创建包含所有平台值的子图
    """
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    # 创建5x2的子图布局
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    fig.suptitle(f'All Platform Comparisons for {user_name}', fontsize=16)
    
    # 将axes转换为一维数组以便遍历
    axes_flat = axes.flatten()
    
    user_data = data[user_name]
    
    for idx, platform_value in enumerate(platform_values):
        ax = axes_flat[idx]
        
        # 准备数据
        plot_data = []
        for exp_type in user_data:
            for trial in user_data[exp_type]:
                if platform_value in user_data[exp_type][trial]:
                    platform_data = user_data[exp_type][trial][platform_value]
                    plot_data.append({
                        'Trial': f'Exp{exp_type}-{trial}',
                        'Median': platform_data['median'],
                        'Q1': platform_data['q1'],
                        'Q3': platform_data['q3'],
                        'Min': platform_data['min'],
                        'Max': platform_data['max']
                    })
        
        if plot_data:
            df = pd.DataFrame(plot_data)
            
            # 绘制箱型图
            box_plot = ax.boxplot(
                [[row['Min'], row['Q1'], row['Median'], row['Q3'], row['Max']]
                 for _, row in df.iterrows()],
                labels=df['Trial'],
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
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存大图
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
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 对数据进行偏移处理
    data = data_offset(data)
    
    # 定义所有可能的平台值
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    # 为每个用户创建单独的文件夹并生成图表
    for user_name in data.keys():
        # 创建用户专属文件夹
        user_fig_path = os.path.join(fig_path, user_name)
        if not os.path.exists(user_fig_path):
            os.makedirs(user_fig_path)
            
        # 生成单独的平台值图表
        for platform_value in platform_values:
            plot_platform_comparison(data, user_name, platform_value, user_fig_path)
            
        # 生成包含所有平台值的大图
        plot_all_platforms_subplots(data, user_name, user_fig_path)
    
    
    
    
