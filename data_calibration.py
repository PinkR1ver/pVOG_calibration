import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import track
from utils import *

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
    
    # 计算误差
    original_error, calibrated_error = calculate_errors(data, calibrated_data)
    
    # 打印结果
    print(f"校准系数: {calibration_coeffs}")
    print(f"原始平均误差: {original_error:.2f}")
    print(f"校准后平均误差: {calibrated_error:.2f}")
    print(f"误差改善: {(original_error - calibrated_error) / original_error * 100:.2f}%")
    
    # 绘制对比图
    plot_calibration_comparison(platform_means, calibrated_means, fig_path)
    
    # 保存校准后的数据
    calibrated_data_path = os.path.join(result_path, "calibrated_statistics_result.json")
    with open(calibrated_data_path, 'w') as f:
        json.dump(calibrated_data, f, indent=4)

if __name__ == "__main__":
    main()