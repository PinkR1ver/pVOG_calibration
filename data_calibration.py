import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import track
from utils import *
import seaborn as sns
from scipy import stats

def calculate_platform_means(data):
    """
    计算每个平台值的所有用户平均值
    返回: {platform_value: [所有用户的平均值列表]}
    """
    platform_means = {}
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    for platform_value in platform_values:
        platform_means[platform_value] = []
        
        for person in data.keys():
            person_platform_data = []
            for exp_type in data[person]:
                for trial in data[person][exp_type]:
                    if platform_value in data[person][exp_type][trial]:
                        platform_data = data[person][exp_type][trial][platform_value]['data']
                        person_platform_data.extend(platform_data)
            
            if person_platform_data:
                platform_means[platform_value].append(np.mean(person_platform_data))
    
    return platform_means

def calculate_calibration_coefficients(platform_means):
    """
    计算校准系数
    理想值: 0_1=0, 0_2=0, ±5=±5, ±10=±10, ±15=±15, ±25=±25
    返回: {platform_value: calibration_coefficient}
    """
    ideal_values = {
        "0_1": 0, "0_2": 0,
        "5": 5, "10": 10, "15": 15, "25": 25,
        "-5": -5, "-10": -10, "-15": -15, "-25": -25
    }
    
    calibration_coeffs = {}
    for platform, means in platform_means.items():
        if platform in ideal_values:
            # 计算平均偏差
            actual_mean = np.mean(means)
            ideal_value = ideal_values[platform]
            
            if abs(actual_mean) > 0.001:  # 避免除以零
                calibration_coeffs[platform] = ideal_value / actual_mean
            else:
                calibration_coeffs[platform] = 1.0
                
    return calibration_coeffs

def apply_calibration(data, calibration_coeffs):
    """
    应用校准系数到原始数据
    返回校准后的数据副本
    """
    calibrated_data = json.loads(json.dumps(data))  # 深拷贝
    
    for person in calibrated_data:
        for exp_type in calibrated_data[person]:
            for trial in calibrated_data[person][exp_type]:
                for platform in calibrated_data[person][exp_type][trial]:
                    if platform in calibration_coeffs:
                        coeff = calibration_coeffs[platform]
                        # 校准所有统计值
                        for key in ['median', 'mean', 'q1', 'q3', 'min', 'max']:
                            calibrated_data[person][exp_type][trial][platform][key] *= coeff
                        # 校准原始数据
                        calibrated_data[person][exp_type][trial][platform]['data'] = [
                            x * coeff for x in calibrated_data[person][exp_type][trial][platform]['data']
                        ]
    
    return calibrated_data

def calculate_errors(data, calibrated_data):
    """
    计算校准前后的误差
    返回: (原始误差, 校准后误差)
    """
    ideal_values = {
        "0_1": 0, "0_2": 0,
        "5": 5, "10": 10, "15": 15, "25": 25,
        "-5": -5, "-10": -10, "-15": -15, "-25": -25
    }
    
    original_errors = []
    calibrated_errors = []
    
    # 计算原始数据误差
    platform_means = calculate_platform_means(data)
    for platform, means in platform_means.items():
        if platform in ideal_values:
            ideal = ideal_values[platform]
            errors = [abs(mean - ideal) for mean in means]
            original_errors.extend(errors)
    
    # 计算校准后数据误差
    calibrated_means = calculate_platform_means(calibrated_data)
    for platform, means in calibrated_means.items():
        if platform in ideal_values:
            ideal = ideal_values[platform]
            errors = [abs(mean - ideal) for mean in means]
            calibrated_errors.extend(errors)
    
    return np.mean(original_errors), np.mean(calibrated_errors)

def plot_calibration_comparison(platform_means, calibrated_means, save_path):
    """
    绘制校准前后的对比图
    """
    plt.figure(figsize=(15, 10))
    
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    ideal_values = {
        "0_1": 0, "0_2": 0,
        "5": 5, "10": 10, "15": 15, "25": 25,
        "-5": -5, "-10": -10, "-15": -15, "-25": -25
    }
    
    # 绘制理想值线
    plt.plot(range(len(platform_order)), 
            [ideal_values[p] for p in platform_order],
            'k--', label='Ideal', linewidth=2)
    
    # 绘制原始平均值
    original_means = [np.mean(platform_means[p]) for p in platform_order]
    plt.plot(range(len(platform_order)), original_means,
             'ro-', label='Original', linewidth=2, markersize=8)
    
    # 绘制校准后平均值
    calibrated_means_values = [np.mean(calibrated_means[p]) for p in platform_order]
    plt.plot(range(len(platform_order)), calibrated_means_values,
             'bo-', label='Calibrated', linewidth=2, markersize=8)
    
    # 添加数值标签
    for i, (orig, cal) in enumerate(zip(original_means, calibrated_means_values)):
        plt.annotate(f'{orig:.2f}', (i, orig), textcoords="offset points",
                    xytext=(0,10), ha='center', color='red')
        plt.annotate(f'{cal:.2f}', (i, cal), textcoords="offset points",
                    xytext=(0,-20), ha='center', color='blue')
    
    plt.title('Platform Values Comparison: Original vs Calibrated',
             pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Platform Value', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.xticks(range(len(platform_order)), platform_order, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'calibration_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def calculate_detailed_errors(data, calibrated_data):
    """
    计算每个实验的校准前后误差
    返回: {
        'original': {person: {exp_type: {trial: error}}},
        'calibrated': {person: {exp_type: {trial: error}}}
    }
    """
    ideal_values = {
        "0_1": 0, "0_2": 0,
        "5": 5, "10": 10, "15": 15, "25": 25,
        "-5": -5, "-10": -10, "-15": -15, "-25": -25
    }
    
    detailed_errors = {
        'original': {},
        'calibrated': {}
    }
    
    # 计算每个实验的误差
    for person in data:
        detailed_errors['original'][person] = {}
        detailed_errors['calibrated'][person] = {}
        
        for exp_type in data[person]:
            detailed_errors['original'][person][exp_type] = {}
            detailed_errors['calibrated'][person][exp_type] = {}
            
            for trial in data[person][exp_type]:
                # 计算原始数据误差
                orig_errors = []
                calib_errors = []
                
                for platform in data[person][exp_type][trial]:
                    if platform in ideal_values:
                        ideal = ideal_values[platform]
                        # 原始数据误差
                        orig_mean = np.mean(data[person][exp_type][trial][platform]['data'])
                        orig_errors.append(abs(orig_mean - ideal))
                        
                        # 校准后数据误差
                        calib_mean = np.mean(calibrated_data[person][exp_type][trial][platform]['data'])
                        calib_errors.append(abs(calib_mean - ideal))
                
                # 保存每个trial的平均误差
                detailed_errors['original'][person][exp_type][trial] = np.mean(orig_errors)
                detailed_errors['calibrated'][person][exp_type][trial] = np.mean(calib_errors)
    
    return detailed_errors

def plot_error_comparison(detailed_errors, save_path):
    """
    绘制校准前后的误差对比图
    """
    plt.figure(figsize=(15, 10))
    
    # 准备数据
    all_orig_errors = []
    all_calib_errors = []
    labels = []
    
    for person in detailed_errors['original']:
        for exp_type in detailed_errors['original'][person]:
            for trial in detailed_errors['original'][person][exp_type]:
                orig_error = detailed_errors['original'][person][exp_type][trial]
                calib_error = detailed_errors['calibrated'][person][exp_type][trial]
                
                all_orig_errors.append(orig_error)
                all_calib_errors.append(calib_error)
                labels.append(f"{person}-Exp{exp_type}-{trial}")
    
    # 绘制散点图
    plt.scatter(range(len(labels)), all_orig_errors, c='red', label='Original Error', alpha=0.6)
    plt.scatter(range(len(labels)), all_calib_errors, c='blue', label='Calibrated Error', alpha=0.6)
    
    # 绘制平均值线
    plt.axhline(y=np.mean(all_orig_errors), color='red', linestyle='--', 
                label=f'Mean Original Error: {np.mean(all_orig_errors):.2f}')
    plt.axhline(y=np.mean(all_calib_errors), color='blue', linestyle='--',
                label=f'Mean Calibrated Error: {np.mean(all_calib_errors):.2f}')
    
    plt.title('Error Comparison Across All Experiments',
             pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Experiments', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'error_comparison_detailed.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def analyze_error_distribution(detailed_errors, save_path):
    """
    分析误差分布并生成可视化
    """
    # 准备数据
    orig_errors = []
    calib_errors = []
    
    for person in detailed_errors['original']:
        for exp_type in detailed_errors['original'][person]:
            for trial in detailed_errors['original'][person][exp_type]:
                orig_errors.append(detailed_errors['original'][person][exp_type][trial])
                calib_errors.append(detailed_errors['calibrated'][person][exp_type][trial])
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Error Distribution Analysis', fontsize=16, y=1.02)
    
    # 1. 直方图和核密度估计
    sns.histplot(data=orig_errors, kde=True, color='red', alpha=0.5, 
                label='Original', ax=ax1)
    sns.histplot(data=calib_errors, kde=True, color='blue', alpha=0.5, 
                label='Calibrated', ax=ax1)
    ax1.set_title('Error Distribution')
    ax1.set_xlabel('Error Value')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # 2. 箱型图
    box_data = pd.DataFrame({
        'Original': orig_errors,
        'Calibrated': calib_errors
    })
    sns.boxplot(data=box_data, ax=ax2)
    ax2.set_title('Error Box Plot')
    ax2.set_ylabel('Error Value')
    
    # 3. Q-Q图
    stats.probplot(orig_errors, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Original)')
    
    stats.probplot(calib_errors, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Calibrated)')
    
    # 计算统计指标
    stats_text = (
        f"Original Errors:\n"
        f"Mean: {np.mean(orig_errors):.2f}\n"
        f"Median: {np.median(orig_errors):.2f}\n"
        f"Std: {np.std(orig_errors):.2f}\n"
        f"IQR: {stats.iqr(orig_errors):.2f}\n"
        f"Skewness: {stats.skew(orig_errors):.2f}\n"
        f"Kurtosis: {stats.kurtosis(orig_errors):.2f}\n\n"
        f"Calibrated Errors:\n"
        f"Mean: {np.mean(calib_errors):.2f}\n"
        f"Median: {np.median(calib_errors):.2f}\n"
        f"Std: {np.std(calib_errors):.2f}\n"
        f"IQR: {stats.iqr(calib_errors):.2f}\n"
        f"Skewness: {stats.skew(calib_errors):.2f}\n"
        f"Kurtosis: {stats.kurtosis(calib_errors):.2f}"
    )
    
    # 添加统计文本到图表
    plt.figtext(1.02, 0.5, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # 执行统计检验
    # Shapiro-Wilk正态性检验
    _, p_orig = stats.shapiro(orig_errors)
    _, p_calib = stats.shapiro(calib_errors)
    # Wilcoxon符号秩检验
    _, p_wilcoxon = stats.wilcoxon(orig_errors, calib_errors)
    
    test_text = (
        f"Statistical Tests:\n"
        f"Shapiro-Wilk test (Original): p={p_orig:.4f}\n"
        f"Shapiro-Wilk test (Calibrated): p={p_calib:.4f}\n"
        f"Wilcoxon test: p={p_wilcoxon:.4f}"
    )
    
    plt.figtext(1.02, 0.2, test_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'error_distribution_analysis.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def analyze_error_by_platform(data, calibrated_data, save_path):
    """
    分析每个平台的误差分布
    """
    ideal_values = {
        "0_1": 0, "0_2": 0,
        "5": 5, "10": 10, "15": 15, "25": 25,
        "-5": -5, "-10": -10, "-15": -15, "-25": -25
    }
    
    platform_errors = {
        'original': {p: [] for p in ideal_values},
        'calibrated': {p: [] for p in ideal_values}
    }
    
    # 收集每个平台的误差
    for person in data:
        for exp_type in data[person]:
            for trial in data[person][exp_type]:
                for platform in data[person][exp_type][trial]:
                    if platform in ideal_values:
                        ideal = ideal_values[platform]
                        # 原始误差
                        orig_mean = np.mean(data[person][exp_type][trial][platform]['data'])
                        platform_errors['original'][platform].append(abs(orig_mean - ideal))
                        # 校准后误差
                        calib_mean = np.mean(calibrated_data[person][exp_type][trial][platform]['data'])
                        platform_errors['calibrated'][platform].append(abs(calib_mean - ideal))
    
    # 创建平台误差对比图
    plt.figure(figsize=(15, 8))
    
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    x = np.arange(len(platform_order))
    width = 0.35
    
    # 计算每个平台的平均误差
    orig_means = [np.mean(platform_errors['original'][p]) for p in platform_order]
    calib_means = [np.mean(platform_errors['calibrated'][p]) for p in platform_order]
    
    # 绘制柱状图
    plt.bar(x - width/2, orig_means, width, label='Original', color='red', alpha=0.6)
    plt.bar(x + width/2, calib_means, width, label='Calibrated', color='blue', alpha=0.6)
    
    # 添加误差线
    orig_stds = [np.std(platform_errors['original'][p]) for p in platform_order]
    calib_stds = [np.std(platform_errors['calibrated'][p]) for p in platform_order]
    plt.errorbar(x - width/2, orig_means, yerr=orig_stds, fmt='none', color='red', alpha=0.3)
    plt.errorbar(x + width/2, calib_means, yerr=calib_stds, fmt='none', color='blue', alpha=0.3)
    
    plt.title('Error Distribution by Platform')
    plt.xlabel('Platform')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(x, platform_order, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'platform_error_distribution.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def calculate_person_platform_means(data, person):
    """
    计算单个用户的每个平台值平均值
    返回: {platform_value: [该用户的平均值列表]}
    """
    platform_means = {}
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    for platform_value in platform_values:
        platform_means[platform_value] = []
        person_platform_data = []
        
        for exp_type in data[person]:
            for trial in data[person][exp_type]:
                if platform_value in data[person][exp_type][trial]:
                    platform_data = data[person][exp_type][trial][platform_value]['data']
                    platform_means[platform_value].append(np.mean(platform_data))
    
    return platform_means

def plot_person_calibration_comparison(original_data, calibrated_data, person, save_path):
    """
    为单个用户绘制每次实验的校准前后对比图
    """
    plt.figure(figsize=(15, 10))
    
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    ideal_values = {
        "0_1": 0, "0_2": 0,
        "5": 5, "10": 10, "15": 15, "25": 25,
        "-5": -5, "-10": -10, "-15": -15, "-25": -25
    }
    
    # 绘制理想值线
    plt.plot(range(len(platform_order)), 
            [ideal_values[p] for p in platform_order],
            'k--', label='Ideal', linewidth=2)
    
    # 计算实验总数
    total_trials = sum(len(trials) for trials in original_data[person].values())
    # 生成足够的颜色
    colors = plt.cm.tab20(np.linspace(0, 1, total_trials))
    color_idx = 0
    
    for exp_type in sorted(original_data[person].keys()):  # 对实验类型排序
        for trial in sorted(original_data[person][exp_type].keys()):  # 对trial排序
            # 收集原始数据点
            original_values = []
            for platform in platform_order:
                if platform in original_data[person][exp_type][trial]:
                    value = np.mean(original_data[person][exp_type][trial][platform]['data'])
                    original_values.append(value)
                else:
                    original_values.append(ideal_values[platform])
            
            # 收集校准后数据点
            calibrated_values = []
            for platform in platform_order:
                if platform in calibrated_data[person][exp_type][trial]:
                    value = np.mean(calibrated_data[person][exp_type][trial][platform]['data'])
                    calibrated_values.append(value)
                else:
                    calibrated_values.append(ideal_values[platform])
            
            # 绘制原始值线
            plt.plot(range(len(platform_order)), original_values,
                    '-o', color=colors[color_idx], 
                    label=f'Original Exp{exp_type}-{trial}',
                    linewidth=2, markersize=8, alpha=0.7)
            
            # 绘制校准后值线
            plt.plot(range(len(platform_order)), calibrated_values,
                    '--o', color=colors[color_idx], 
                    label=f'Calibrated Exp{exp_type}-{trial}',
                    linewidth=2, markersize=8, alpha=0.7)
            
            # 添加数值标签
            for i, (orig, cal) in enumerate(zip(original_values, calibrated_values)):
                plt.annotate(f'{orig:.2f}', (i, orig), textcoords="offset points",
                            xytext=(0,10), ha='center', color=colors[color_idx], fontsize=8)
                plt.annotate(f'{cal:.2f}', (i, cal), textcoords="offset points",
                            xytext=(0,-20), ha='center', color=colors[color_idx], fontsize=8)
            
            color_idx += 1
    
    plt.title(f'Platform Values Comparison for {person}: Original vs Calibrated',
             fontsize=14, fontweight='bold')
    plt.xlabel('Platform Value', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.xticks(range(len(platform_order)), platform_order, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f'{person}_calibration_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_single_experiment_comparison(original_data, calibrated_data, person, exp_type, trial, save_path):
    """
    绘制单次实验的校准对比图
    """
    plt.figure(figsize=(15, 10))
    
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    ideal_values = {
        "0_1": 0, "0_2": 0,
        "5": 5, "10": 10, "15": 15, "25": 25,
        "-5": -5, "-10": -10, "-15": -15, "-25": -25
    }
    
    # 绘制理想值线
    plt.plot(range(len(platform_order)), 
            [ideal_values[p] for p in platform_order],
            'k--', label='Ideal', linewidth=2)
    
    # 收集数据点
    original_values = []
    calibrated_values = []
    for platform in platform_order:
        if platform in original_data[person][exp_type][trial]:
            orig_value = np.mean(original_data[person][exp_type][trial][platform]['data'])
            calib_value = np.mean(calibrated_data[person][exp_type][trial][platform]['data'])
            original_values.append(orig_value)
            calibrated_values.append(calib_value)
        else:
            original_values.append(ideal_values[platform])
            calibrated_values.append(ideal_values[platform])
    
    # 绘制原始值线（红色）
    plt.plot(range(len(platform_order)), original_values,
            '-o', color='#FF4B4B', 
            label='Original',
            linewidth=2, markersize=8, alpha=0.8)
    
    # 绘制校准后值线（蓝色）
    plt.plot(range(len(platform_order)), calibrated_values,
            '-o', color='#4B7BFF', 
            label='Calibrated',
            linewidth=2, markersize=8, alpha=0.8)
    
    # 添加数值标签
    for i, (orig, cal) in enumerate(zip(original_values, calibrated_values)):
        plt.annotate(f'{orig:.2f}', (i, orig), textcoords="offset points",
                    xytext=(0,10), ha='center', color='#FF4B4B', fontsize=10)
        plt.annotate(f'{cal:.2f}', (i, cal), textcoords="offset points",
                    xytext=(0,-20), ha='center', color='#4B7BFF', fontsize=10)
    
    plt.title(f'Platform Values Comparison for {person}\nExp{exp_type}-{trial}',
             fontsize=14, fontweight='bold')
    plt.xlabel('Platform Value', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.xticks(range(len(platform_order)), platform_order, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f'{person}_exp{exp_type}_trial{trial}_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_all_experiments_comparison(original_data, calibrated_data, person, save_path):
    """
    绘制所有实验的对比图
    """
    plt.figure(figsize=(15, 10))
    
    platform_order = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    ideal_values = {
        "0_1": 0, "0_2": 0,
        "5": 5, "10": 10, "15": 15, "25": 25,
        "-5": -5, "-10": -10, "-15": -15, "-25": -25
    }
    
    # 绘制理想值线
    plt.plot(range(len(platform_order)), 
            [ideal_values[p] for p in platform_order],
            'k--', label='Ideal', linewidth=2)
    
    # 为每个实验类型选择不同的颜色
    exp_colors = plt.cm.tab20(np.linspace(0, 1, len(original_data[person])))
    
    for idx, exp_type in enumerate(sorted(original_data[person].keys())):
        for trial in sorted(original_data[person][exp_type].keys()):
            # 收集数据点
            original_values = []
            calibrated_values = []
            for platform in platform_order:
                if platform in original_data[person][exp_type][trial]:
                    orig_value = np.mean(original_data[person][exp_type][trial][platform]['data'])
                    calib_value = np.mean(calibrated_data[person][exp_type][trial][platform]['data'])
                    original_values.append(orig_value)
                    calibrated_values.append(calib_value)
                else:
                    original_values.append(ideal_values[platform])
                    calibrated_values.append(ideal_values[platform])
            
            # 绘制原始值线
            plt.plot(range(len(platform_order)), original_values,
                    '-o', color=exp_colors[idx], 
                    label=f'Original Exp{exp_type}-{trial}',
                    linewidth=2, markersize=8, alpha=0.7)
            
            # 绘制校准后值线（统一使用蓝色，但透明度不同）
            plt.plot(range(len(platform_order)), calibrated_values,
                    '--o', color='#4B7BFF', 
                    label=f'Calibrated Exp{exp_type}-{trial}',
                    linewidth=2, markersize=8, alpha=0.5)
    
    plt.title(f'Platform Values Comparison for {person}: All Experiments',
             fontsize=14, fontweight='bold')
    plt.xlabel('Platform Value', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.xticks(range(len(platform_order)), platform_order, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f'{person}_all_experiments_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(base_path, "result")
    data_path = os.path.join(result_path, "statistics_result.json")
    
    fig_path = os.path.join(base_path, "fig", "calibration_analysis")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    # 加载数据
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 对数据进行偏移处理
    data = data_offset(data)
    
    # 计算平台平均值
    platform_means = calculate_platform_means(data)
    
    # 计算校准系数
    calibration_coeffs = calculate_calibration_coefficients(platform_means)
    
    # 应用校准
    calibrated_data = apply_calibration(data, calibration_coeffs)
    
    # 计算校准后的平台平均值
    calibrated_means = calculate_platform_means(calibrated_data)
    
    # 计算详细误差
    detailed_errors = calculate_detailed_errors(data, calibrated_data)
    
    # 计算总体平均误差
    original_error = np.mean([
        error
        for person in detailed_errors['original']
        for exp_type in detailed_errors['original'][person]
        for error in detailed_errors['original'][person][exp_type].values()
    ])
    
    calibrated_error = np.mean([
        error
        for person in detailed_errors['calibrated']
        for exp_type in detailed_errors['calibrated'][person]
        for error in detailed_errors['calibrated'][person][exp_type].values()
    ])
    
    # 打印结果
    print(f"校准系数: {calibration_coeffs}")
    print(f"原始平均误差: {original_error:.2f}")
    print(f"校准后平均误差: {calibrated_error:.2f}")
    print(f"误差改善: {(original_error - calibrated_error) / original_error * 100:.2f}%")
    
    # 保存详细误差数据
    error_data_path = os.path.join(result_path, "calibration_errors.json")
    with open(error_data_path, 'w') as f:
        json.dump(detailed_errors, f, indent=4)
    
    # 为每个用户创建单独的文件夹并生成校准对比图
    for person in track(data.keys(), description="Generating calibration plots for each person"):
        # 创建用户专属文件夹
        person_fig_path = os.path.join(fig_path, person)
        if not os.path.exists(person_fig_path):
            os.makedirs(person_fig_path)
        
        # 生成每次实验的单独对比图
        for exp_type in data[person]:
            for trial in data[person][exp_type]:
                plot_single_experiment_comparison(
                    data, calibrated_data, person, exp_type, trial, person_fig_path)
        
        # 生成所有实验的总览图
        plot_all_experiments_comparison(data, calibrated_data, person, person_fig_path)

if __name__ == "__main__":
    main()