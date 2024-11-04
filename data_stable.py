import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def calculate_cv(data):
    """计算变异系数 (CV = 标准差/平均值 * 100%)"""
    return np.std(data) / np.abs(np.mean(data)) * 100

def analyze_stability(data_path):
    # 读取数据
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 存储每个用户在每个平台角度下的平均值和中值
    platform_stats = {}  # 平台角度 -> 用户 -> (平均值, 中值)
    user_trial_stats = {}  # 用户 -> 平台角度 -> 试验数据列表
    
    # 遍历每个用户
    for user in data:
        for exp_type in data[user].keys():
            user_data = data[user][exp_type]
            user_trial_stats[user] = {}
            
            # 遍历每个试验
            for trial in user_data:
                for platform in user_data[trial]:
                    if '_' not in platform:
                        platform_data = user_data[trial][platform]['data']
                        
                        # 为整体稳定性存储数据
                        if platform not in platform_stats:
                            platform_stats[platform] = {}
                        if user not in platform_stats[platform]:
                            platform_stats[platform][user] = []
                        
                        # 为个人稳定性存储数据
                        if platform not in user_trial_stats[user]:
                            user_trial_stats[user][platform] = []
                        
                        # 计算这次试验的平均值和中值
                        mean_val = np.mean(platform_data)
                        median_val = np.median(platform_data)
                        
                        platform_stats[platform][user].append((mean_val, median_val))
                        user_trial_stats[user][platform].append((mean_val, median_val))
    
    # 1. 计算每个平台角度的整体稳定性
    platform_cv = {}
    for platform in platform_stats:
        user_mean_cvs = []  # 存储每个用户的平均值CV
        user_median_cvs = []  # 存储每个用户的中值CV
        
        for user in platform_stats[platform]:
            trials = platform_stats[platform][user]
            user_means = [trial[0] for trial in trials]
            user_medians = [trial[1] for trial in trials]
            
            # 计算每个用户的CV
            user_mean_cv = calculate_cv(user_means)
            user_median_cv = calculate_cv(user_medians)
            
            user_mean_cvs.append(user_mean_cv)
            user_median_cvs.append(user_median_cv)
        
        # 计算所有用户CV的平均值
        platform_cv[platform] = {
            'mean_cv': np.mean(user_mean_cvs),
            'median_cv': np.mean(user_median_cvs)
        }
    
    # 2. 计算每个用户在每个平台角度的稳定性
    user_cv = {}
    for user in user_trial_stats:
        user_cv[user] = {}
        for platform in user_trial_stats[user]:
            trials = user_trial_stats[user][platform]
            trial_means = [trial[0] for trial in trials]
            trial_medians = [trial[1] for trial in trials]
            
            user_cv[user][platform] = {
                'mean_cv': calculate_cv(trial_means),
                'median_cv': calculate_cv(trial_medians)
            }
    
    # 创建结果DataFrame
    platform_cv_df = pd.DataFrame(platform_cv).T.round(2)
    
    # 创建用户CV的DataFrame
    user_cv_data = {
        user: {
            platform: data['mean_cv'] 
            for platform, data in platforms.items()
        }
        for user, platforms in user_cv.items()
    }
    user_cv_df = pd.DataFrame(user_cv_data).round(2)
    
    return platform_cv_df, user_cv_df

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_cv_comparison(cv_df):
    """绘制平台角度的整体稳定性对比图"""
    plt.figure(figsize=(10, 6))
    
    platform_order = ["-25", "-15", "-10", "-5", "5", "10", "15", "25"]
    cv_df = cv_df.reindex(platform_order)
    
    x = np.arange(len(platform_order))
    width = 0.35
    
    plt.bar(x - width/2, cv_df['mean_cv'], width, label='Mean CV')
    plt.bar(x + width/2, cv_df['median_cv'], width, label='Median CV')
    
    plt.xlabel('Platform Angles')
    plt.ylabel('CV (%)')
    plt.title('Overall Stability Comparison: Mean vs Median')
    plt.xticks(x, platform_order)
    plt.legend()
    
    plt.tight_layout()
    ensure_dir('fig/stable')
    plt.savefig('fig/stable/platform_cv_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_cv_heatmap(user_cv_df):
    """绘制用户在各平台角度的稳定性热图"""
    plt.figure(figsize=(12, 8))
    
    # 转置DataFrame使平台角度作为列
    user_cv_df = user_cv_df.T
    
    # 重新排序平台角度
    platform_order = ["5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    user_cv_df = user_cv_df[platform_order]
    
    sns.heatmap(user_cv_df, 
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': 'CV (%)'})
    
    plt.title('Individual Stability Analysis (CV %)')
    plt.xlabel('Platform Angles')
    plt.ylabel('Users')
    
    plt.tight_layout()
    ensure_dir('fig/stable')
    plt.savefig('fig/stable/user_cv_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data_path = "result/statistics_result.json"
    
    # 分析稳定性
    platform_cv_df, user_cv_df = analyze_stability(data_path)
    
    # 打印结果
    print("\n平台角度整体稳定性分析 (CV %):")
    print(platform_cv_df)
    print("\n个人稳定性分析 (CV %):")
    print(user_cv_df)
    
    # 绘制对比图
    plot_cv_comparison(platform_cv_df)
    plot_user_cv_heatmap(user_cv_df)
    
    # 保存结果到Excel
    ensure_dir('fig/stable')
    with pd.ExcelWriter('fig/stable/stability_analysis.xlsx') as writer:
        platform_cv_df.to_excel(writer, sheet_name='Platform_Stability')
        user_cv_df.to_excel(writer, sheet_name='User_Stability')

if __name__ == "__main__":
    main()