import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from rich.progress import track
from utils import *
import pandas as pd
from matplotlib.gridspec import GridSpec

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def fit_calibration_curve(measured, target, degree=1):
    """
    拟合校准曲线
    degree: 多项式阶数，1为线性拟合
    """
    # 准备数据
    X = np.array(measured).reshape(-1, 1)
    y = np.array(target)
    
    if degree == 1:
        # 线性回归
        slope, intercept, r_value, p_value, std_err = linregress(measured, target)
        return lambda x: slope * x + intercept, {
            'slope': slope,
            'intercept': intercept,
            'r2': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }
    else:
        # 多项式回归
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        return lambda x: model.predict(poly_features.transform(np.array(x).reshape(-1, 1))), {
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'r2': r2_score(y, model.predict(X_poly))
        }

def analyze_calibration(data, save_dir, degree=1):
    """分析校准效果"""
    # 存储所有数据
    all_data = {
        'individual': {},  # 个人数据
        'overall': {      # 整体数据
            'measured': [],
            'target': []
        }
    }
    
    # 收集数据，遍历两种实验类型
    for user in data:
        all_data['individual'][user] = {
            'measured': [],
            'target': []
        }
        
        # 遍历实验类型1和2
        for exp_type in ['1', '2']:
            if exp_type in data[user]:  # 确保实验类型存在
                user_data = data[user][exp_type]
                for trial in user_data:
                    for platform in user_data[trial]:
                        if '_' not in platform:
                            measured = np.mean(user_data[trial][platform]['data'])
                            target = float(platform)
                            
                            # 添加到个人数据
                            all_data['individual'][user]['measured'].append(measured)
                            all_data['individual'][user]['target'].append(target)
                            
                            # 添加到整体数据
                            all_data['overall']['measured'].append(measured)
                            all_data['overall']['target'].append(target)
    
    # 分析结果
    results = {
        'individual': {},
        'overall': {}
    }
    
    # 1. 分析整体数据
    print("分析整体校准效果...")
    calib_func, stats = fit_calibration_curve(
        all_data['overall']['measured'],
        all_data['overall']['target'],
        degree
    )
    
    calibrated = calib_func(np.array(all_data['overall']['measured']))
    
    results['overall'] = {
        'stats': stats,
        'original_error': np.abs(np.array(all_data['overall']['measured']) - 
                               np.array(all_data['overall']['target'])),
        'calibrated_error': np.abs(calibrated - np.array(all_data['overall']['target']))
    }
    
    # 绘制整体校准结果
    plot_calibration_results(
        all_data['overall']['measured'],
        calibrated,
        all_data['overall']['target'],
        'Overall',
        os.path.join(save_dir, 'overall_calibration.png'),
        stats
    )
    
    # 2. 分析个人数据
    print("分析个人校准效果...")
    for user in track(all_data['individual'], description="Processing users"):
        calib_func, stats = fit_calibration_curve(
            all_data['individual'][user]['measured'],
            all_data['individual'][user]['target'],
            degree
        )
        
        calibrated = calib_func(np.array(all_data['individual'][user]['measured']))
        
        results['individual'][user] = {
            'stats': stats,
            'original_error': np.abs(np.array(all_data['individual'][user]['measured']) - 
                                   np.array(all_data['individual'][user]['target'])),
            'calibrated_error': np.abs(calibrated - np.array(all_data['individual'][user]['target']))
        }
        
        # 绘制个人校准结果
        plot_calibration_results(
            all_data['individual'][user]['measured'],
            calibrated,
            all_data['individual'][user]['target'],
            f'User: {user}',
            os.path.join(save_dir, f'{user}_calibration.png'),
            stats
        )
    
    return results, all_data

def plot_calibration_results(measured, calibrated, target, title, save_path, stats):
    """绘制校准结果"""
    plt.figure(figsize=(15, 5))
    
    # 1. 校准效果散点图
    plt.subplot(1, 3, 1)
    
    # 绘制原始数据点和拟合线
    plt.scatter(measured, target, alpha=0.5, label='Original Data', color='blue')
    
    # 添加理想线（y=x）
    ideal_line = np.linspace(min(min(measured), min(target)), 
                           max(max(measured), max(target)), 100)
    plt.plot(ideal_line, ideal_line, 'k--', label='Ideal Line')
    
    # 添加拟合线和校准后的数据点
    if 'slope' in stats:  # 线性拟合
        x_fit = np.linspace(min(measured), max(measured), 100)
        y_fit = stats['slope'] * x_fit + stats['intercept']
        plt.plot(x_fit, y_fit, 'r-', 
                label=f'Fit: y={stats["slope"]:.2f}x+{stats["intercept"]:.2f}')
        # 添加校准后的数据点
        plt.scatter(measured, calibrated, alpha=0.5, label='Calibrated', color='red')
    
    plt.xlabel('Measured Angle')
    plt.ylabel('Target/Calibrated Angle')
    plt.title(f'Calibration Results\n{title}\nR² = {stats["r2"]:.3f}')
    plt.legend()
    
    # 2. 误差箱线图
    plt.subplot(1, 3, 2)
    original_error = np.abs(np.array(measured) - np.array(target))
    calibrated_error = np.abs(np.array(calibrated) - np.array(target))
    
    plt.boxplot([original_error, calibrated_error],
                labels=['Original Error', 'Calibrated Error'])
    plt.ylabel('Absolute Error')
    plt.title('Error Distribution')
    
    # 3. Bland-Altman图
    plt.subplot(1, 3, 3)
    mean_orig = (np.array(measured) + np.array(target)) / 2
    diff_orig = np.array(measured) - np.array(target)
    plt.scatter(mean_orig, diff_orig, alpha=0.5, label='Original')
    
    mean_cal = (np.array(calibrated) + np.array(target)) / 2
    diff_cal = np.array(calibrated) - np.array(target)
    plt.scatter(mean_cal, diff_cal, alpha=0.5, label='Calibrated')
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Mean of Measured and Target')
    plt.ylabel('Difference (Measured - Target)')
    plt.title('Bland-Altman Plot')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_comparison(results, save_dir):
    """绘制误差对比图"""
    # 1. 整体误差对比
    plt.figure(figsize=(10, 6))
    data = {
        'Original': results['overall']['original_error'],
        'Calibrated': results['overall']['calibrated_error']
    }
    
    sns.boxplot(data=data)
    plt.ylabel('Absolute Error')
    plt.title('Overall Error Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_error_comparison.png'), dpi=300)
    plt.close()
    
    # 2. 个人误差对比
    plt.figure(figsize=(15, 8))
    individual_data = {
        'Original': [],
        'Calibrated': [],
        'User': []
    }
    
    for user in results['individual']:
        individual_data['Original'].extend(results['individual'][user]['original_error'])
        individual_data['Calibrated'].extend(results['individual'][user]['calibrated_error'])
        individual_data['User'].extend([user] * len(results['individual'][user]['original_error']))
    
    sns.boxplot(x='User', y='value', hue='variable',
                data=pd.melt(pd.DataFrame(individual_data), 
                            id_vars=['User'],
                            value_vars=['Original', 'Calibrated']))
    
    plt.ylabel('Absolute Error')
    plt.title('Individual Error Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'individual_error_comparison.png'), dpi=300)
    plt.close()

def plot_calibration_comparison(measured, target, degrees, save_path):
    """对比不同阶数多项式拟合的效果"""
    plt.figure(figsize=(15, 15))
    
    # 创建GridSpec来更好地控制子图布局
    gs = GridSpec(3, 2, figure=plt.gcf())
    
    results = {}
    
    # 1-4: 不同阶数的拟合效果图
    for i, degree in enumerate(degrees, 1):
        plt.subplot(gs[i-1]) if i <= 2 else plt.subplot(gs[i-1])
        
        # 拟合模型
        calib_func, stats = fit_calibration_curve(measured, target, degree)
        calibrated = calib_func(np.array(measured))
        results[degree] = {'stats': stats, 'calibrated': calibrated}
        
        # 绘制原始数据点和校准后的数据点
        plt.scatter(measured, target, alpha=0.5, label='Original', color='blue')
        plt.scatter(measured, calibrated, alpha=0.5, label='Calibrated', color='red')
        
        # 添加拟合线
        x_fit = np.linspace(min(measured), max(measured), 100)
        y_fit = calib_func(x_fit)
        plt.plot(x_fit, y_fit, 'g-', label='Calibration Function')
        
        plt.xlabel('Measured Angle')
        plt.ylabel('Target/Calibrated Angle')
        plt.title(f'Degree {degree} Polynomial\nR² = {stats["r2"]:.3f}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
    
    # 5: 误差箱线图
    plt.subplot(gs[4])
    error_data = []
    labels = []
    
    # 原始误差
    original_error = np.abs(np.array(measured) - np.array(target))
    error_data.append(original_error)
    labels.append('Original')
    
    # 不同阶数的校准后误差
    for degree in degrees:
        calibrated_error = np.abs(results[degree]['calibrated'] - np.array(target))
        error_data.append(calibrated_error)
        labels.append(f'Degree {degree}')
    
    plt.boxplot(error_data, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel('Absolute Error (degrees)')
    plt.title('Error Distribution Comparison')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 6: 合并的Bland-Altman图
    plt.subplot(gs[5])
    colors = ['blue', 'red', 'green', 'purple']  # 不同阶数使用不同颜色
    
    for degree, color in zip(degrees, colors):
        calibrated = results[degree]['calibrated']
        differences = calibrated - np.array(target)
        means = (calibrated + np.array(target)) / 2
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        plt.scatter(means, differences, alpha=0.5, color=color, 
                   label=f'Degree {degree}')
        plt.axhline(y=mean_diff, color=color, linestyle='-', alpha=0.3)
        plt.axhline(y=mean_diff + 1.96*std_diff, color=color, linestyle='--', alpha=0.3)
        plt.axhline(y=mean_diff - 1.96*std_diff, color=color, linestyle='--', alpha=0.3)
    
    plt.xlabel('Mean of Calibrated and Target (degrees)')
    plt.ylabel('Difference (Calibrated - Target) (degrees)')
    plt.title('Bland-Altman Plot Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成并保存统计信息
    stats_text = "Statistical Summary:\n\n"
    
    # 原始误差统计
    stats_text += "Original Error:\n"
    stats_text += f"Mean ± SD: {np.mean(original_error):.2f}° ± {np.std(original_error):.2f}°\n"
    stats_text += f"Median (IQR): {np.median(original_error):.2f}° "
    stats_text += f"({np.percentile(original_error, 25):.2f}° - {np.percentile(original_error, 75):.2f}°)\n\n"
    
    # 各阶数校准后的误差统计
    for degree in degrees:
        calibrated_error = np.abs(results[degree]['calibrated'] - np.array(target))
        stats_text += f"Degree {degree} Error:\n"
        stats_text += f"Mean ± SD: {np.mean(calibrated_error):.2f}° ± {np.std(calibrated_error):.2f}°\n"
        stats_text += f"Median (IQR): {np.median(calibrated_error):.2f}° "
        stats_text += f"({np.percentile(calibrated_error, 25):.2f}° - {np.percentile(calibrated_error, 75):.2f})°\n\n"
    
    # 保存统计信息到文本文件
    stats_path = os.path.splitext(save_path)[0] + '_stats.txt'
    with open(stats_path, 'w') as f:
        f.write(stats_text)
    
    return results

def analyze_overall_calibration_effect(data, degrees, save_dir):
    """
    分析使用整体拟合曲线应用到每个人数据的效果
    """
    # 收集所有数据和个人数据
    all_measured = []
    all_target = []
    individual_data = {}
    
    for user in data:
        individual_data[user] = {
            'measured': [],
            'target': []
        }
        user_data = data[user]['1']
        
        for trial in user_data:
            for platform in user_data[trial]:
                if '_' not in platform:
                    measured = np.mean(user_data[trial][platform]['data'])
                    target = float(platform)
                    
                    all_measured.append(measured)
                    all_target.append(target)
                    individual_data[user]['measured'].append(measured)
                    individual_data[user]['target'].append(target)
    
    # 对每个阶数进行拟合和分析
    overall_results = {}
    
    for degree in degrees:
        # 使用所有数据拟合模型
        calib_func, stats = fit_calibration_curve(all_measured, all_target, degree)
        
        # 存储每个用户使用整体模型的校准效果
        user_results = {}
        error_data = {
            'Original': [],
            'Calibrated': [],
            'User': []
        }
        
        for user in individual_data:
            measured = individual_data[user]['measured']
            target = individual_data[user]['target']
            
            # 使用整体模型进行校准
            calibrated = calib_func(np.array(measured))
            
            # 计算误差
            original_error = np.abs(np.array(measured) - np.array(target))
            calibrated_error = np.abs(calibrated - np.array(target))
            
            user_results[user] = {
                'original_error': original_error.tolist(),  # 转换为列表
                'calibrated_error': calibrated_error.tolist(),  # 转换为列表
                'error_reduction': float(np.mean(original_error) - np.mean(calibrated_error))  # 转换为float
            }
            
            # 收集误差数据用于绘图
            error_data['Original'].extend(original_error)
            error_data['Calibrated'].extend(calibrated_error)
            error_data['User'].extend([user] * len(original_error))
        
        overall_results[degree] = {
            'stats': stats,
            'user_results': user_results
        }
        
        # 绘制使用整体模型的误差对比图
        plt.figure(figsize=(15, 8))
        df = pd.DataFrame(error_data)
        sns.boxplot(x='User', y='value', hue='variable',
                   data=pd.melt(df, id_vars=['User'],
                               value_vars=['Original', 'Calibrated']))
        
        plt.ylabel('Absolute Error')
        plt.title(f'Individual Error Comparison Using Overall Model (Degree {degree})')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        ensure_dir(os.path.join(save_dir, 'overall_model_effect'))
        plt.savefig(os.path.join(save_dir, 'overall_model_effect', 
                                f'degree_{degree}_error_comparison.png'), 
                   dpi=300)
        plt.close()
    
    return overall_results

def collect_all_data(data):
    """收集所有用户的数据"""
    measured = []
    target = []
    
    for user in data:
        # 遍历实验类型1和2
        for exp_type in ['1', '2']:
            if exp_type in data[user]:  # 确保实验类型存在
                user_data = data[user][exp_type]
                for trial in user_data:
                    for platform in user_data[trial]:
                        if '_' not in platform:  # 跳过特殊平台数据
                            measured.append(np.mean(user_data[trial][platform]['data']))
                            target.append(float(platform))
    
    return measured, target

def collect_user_data(data, user):
    """收集单个用户的数据"""
    measured = []
    target = []
    
    # 遍历实验类型1和2
    for exp_type in ['1', '2']:
        if exp_type in data[user]:  # 确保实验类型存在
            user_data = data[user][exp_type]
            for trial in user_data:
                for platform in user_data[trial]:
                    if '_' not in platform:  # 跳过特殊平台数据
                        measured.append(np.mean(user_data[trial][platform]['data']))
                        target.append(float(platform))
    
    return measured, target

def serialize_and_save_results(results, save_dir):
    """将结果转换为可JSON序列化的格式并保存"""
    def convert_to_serializable(obj):
        """递归转换所有numpy数组为列表"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    # 转换所有结果
    serializable_results = convert_to_serializable(results)
    
    # 保存结果
    with open(os.path.join(save_dir, 'polynomial_comparison_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)

def main():
    # 读取数据
    with open('result/statistics_result.json', 'r') as f:
        data = json.load(f)
        
    data = data_offset(data)
    
    # 创建保存目录
    save_dir = 'fig/regression_calibration'
    ensure_dir(save_dir)
    
    # 对比不同阶数的拟合效果
    degrees = [1, 2, 3, 4]  # 对比1-4阶多项式
    
    # 1. 整体数据的对比
    print("分析整体校准效果...")
    measured, target = collect_all_data(data)
    overall_results = plot_calibration_comparison(
        measured, target, degrees,
        os.path.join(save_dir, 'overall_polynomial_comparison.png')
    )
    
    # 2. 个人数据的对比
    print("分析个人校准效果...")
    individual_results = {}
    for user in track(data, description="Processing users"):
        # 创建用户专属目录
        user_dir = os.path.join(save_dir, 'individual', user)
        ensure_dir(user_dir)
        
        user_measured, user_target = collect_user_data(data, user)
        individual_results[user] = plot_calibration_comparison(
            user_measured, user_target, degrees,
            os.path.join(user_dir, 'polynomial_comparison.png')
        )
    
    # 3. 分析整体模型应用到个人数据的效果
    print("分析整体模型在个人数据上的效果...")
    overall_model_results = analyze_overall_calibration_effect(
        data, degrees, save_dir
    )
    
    # 保存所有结果
    results = {
        'overall': overall_results,
        'individual': individual_results,
        'overall_model_effect': overall_model_results
    }
    
    # 将结果转换为可JSON序列化的格式并保存
    serialize_and_save_results(results, save_dir)
    
    # 生成汇总报告
    generate_summary_report(results, save_dir)

def generate_summary_report(results, save_dir):
    """生成汇总报告"""
    report = []
    report.append("# 校准分析报告\n")
    
    # 1. 整体模型分析
    report.append("## 整体模型分析")
    for degree in results['overall']:
        r2 = results['overall'][degree]['stats']['r2']
        report.append(f"\n### {degree}阶多项式")
        report.append(f"- R² 值: {r2:.4f}")
    
    # 2. 个人模型分析
    report.append("\n## 个人模型分析")
    for user in results['individual']:
        report.append(f"\n### 用户: {user}")
        for degree in results['individual'][user]:
            r2 = results['individual'][user][degree]['stats']['r2']
            report.append(f"- {degree}阶多项式 R²: {r2:.4f}")
    
    # 保存报告
    with open(os.path.join(save_dir, 'calibration_report.md'), 'w', 
              encoding='utf-8') as f:
        f.write('\n'.join(report))

if __name__ == "__main__":
    main()