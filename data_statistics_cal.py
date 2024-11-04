# 在这个文件里，我们要把result里的所有人名文件下的json文件整合在一起，同时不需要他们每次平台的数值series，而是计算出这串数值的平均值
# 和中值保存下来

import os
import json
import numpy as np
import re


def remove_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 0.2 * iqr
    upper_bound = q3 + 0.2 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def process_platform_data(file_path):
    """处理平台数据文件"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 获取所有平台数据
    platforms = data['experiments'][exp_type][trial_num]
    results = {}
    
    # 处理每个平台的数据
    for platform, values in platforms.items():
        # 跳过非数值型平台(比如"name"等)
        if not isinstance(values, list):
            continue
            
        # 去除异常值
        cleaned_data = remove_outliers_iqr(values)
        
        # 计算统计值
        results[platform] = {
            'median': np.median(cleaned_data),
            'mean': np.mean(cleaned_data),
            'q1': np.percentile(cleaned_data, 25),
            'q3': np.percentile(cleaned_data, 75),
            'min': np.min(cleaned_data),
            'max': np.max(cleaned_data),
            'original_length': len(values),
            'cleaned_length': len(cleaned_data),
            'data': cleaned_data
        }
    
    return results   


if __name__ == "__main__":
    
    base_path = os.path.dirname(__file__)
    result_path = os.path.join(base_path, "result")
    final_results = {}

    for root, dirs, files in os.walk(result_path):
        
        relative_path = os.path.relpath(root, result_path)
        if relative_path == ".":
            continue
        level = len(relative_path.split(os.sep))
        
        if level == 1:
            
            user_name = os.path.basename(root)
            final_results[user_name] = {}
            
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    exp_type = file.split("_")[0].split("-")[0]
                    trial_num = file.split("_")[0].split("-")[1]
                    
                    if exp_type not in final_results[user_name]:
                        final_results[user_name][exp_type] = {}
                        
                    if trial_num not in final_results[user_name][exp_type]:
                        final_results[user_name][exp_type][trial_num] = {}
                    
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        
                    platform_data = process_platform_data(file_path)
                    final_results[user_name][exp_type][trial_num] = platform_data
                    
    output_path = os.path.join(result_path, "statistics_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
                    
                


