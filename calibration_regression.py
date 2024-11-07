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
from sklearn.model_selection import KFold

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

def fit_and_evaluate_with_cv(measured, target, degree, n_splits=5):
    """使用交叉验证拟合和评估校准模型"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {
        'train_errors': [],
        'test_errors': [],
        'train_r2': [],
        'test_r2': []
    }
    
    measured = np.array(measured)
    target = np.array(target)
    
    for train_idx, test_idx in kf.split(measured):
        # 分割数据
        X_train, X_test = measured[train_idx], measured[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]
        
        # 拟合模型
        calib_func, stats = fit_calibration_curve(X_train, y_train, degree)
        
        # 预测
        y_train_pred = calib_func(X_train)
        y_test_pred = calib_func(X_test)
        
        # 计算误差
        train_error = np.mean(np.abs(y_train_pred - y_train))
        test_error = np.mean(np.abs(y_test_pred - y_test))
        
        # 计算R²
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 存储结果
        cv_results['train_errors'].append(train_error)
        cv_results['test_errors'].append(test_error)
        cv_results['train_r2'].append(train_r2)
        cv_results['test_r2'].append(test_r2)
    
    return cv_results

def plot_calibration_comparison(measured, target, degrees, save_path):
    """对比不同阶数多项式拟合的效果"""
    plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 2, figure=plt.gcf())
    results = {}
    
    # 添加交叉验证结果存储
    cv_results = {}
    
    # 1-4: 不同阶数的拟合效果图
    for i, degree in enumerate(degrees, 1):
        plt.subplot(gs[i-1]) if i <= 2 else plt.subplot(gs[i-1])
        
        # 拟合模型和交叉验证
        calib_func, stats = fit_calibration_curve(measured, target, degree)
        cv_result = fit_and_evaluate_with_cv(measured, target, degree)
        cv_results[degree] = cv_result
        
        calibrated = calib_func(np.array(measured))
        results[degree] = {'stats': stats, 'calibrated': calibrated}
        
        # 绘制数据点和拟合线
        plt.scatter(measured, target, alpha=0.5, label='Original', color='blue')
        plt.scatter(measured, calibrated, alpha=0.5, label='Calibrated', color='red')
        
        x_fit = np.linspace(min(measured), max(measured), 100)
        y_fit = calib_func(x_fit)
        plt.plot(x_fit, y_fit, 'g-', label='Calibration Function')
        
        # 添加交叉验证结果到标题
        mean_train_error = np.mean(cv_result['train_errors'])
        mean_test_error = np.mean(cv_result['test_errors'])
        plt.title(f'Degree {degree} Polynomial\n' + 
                 f'R² = {stats["r2"]:.3f}\n' +
                 f'CV Train Error: {mean_train_error:.2f}°\n' +
                 f'CV Test Error: {mean_test_error:.2f}°')
        
        plt.xlabel('Measured Angle')
        plt.ylabel('Target/Calibrated Angle')
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
    
    # 更新统计信息文本
    stats_text = "Statistical Summary:\n\n"
    stats_text += "Cross-Validation Results:\n"
    for degree in degrees:
        cv_result = cv_results[degree]
        stats_text += f"\nDegree {degree}:\n"
        stats_text += f"Train Error (Mean ± SD): {np.mean(cv_result['train_errors']):.2f}° ± {np.std(cv_result['train_errors']):.2f}°\n"
        stats_text += f"Test Error (Mean ± SD): {np.mean(cv_result['test_errors']):.2f}° ± {np.std(cv_result['test_errors']):.2f}°\n"
        stats_text += f"Train R² (Mean ± SD): {np.mean(cv_result['train_r2']):.3f} ± {np.std(cv_result['train_r2']):.3f}\n"
        stats_text += f"Test R² (Mean ± SD): {np.mean(cv_result['test_r2']):.3f} ± {np.std(cv_result['test_r2']):.3f}\n"
    
    stats_text +=  "\n\n"
    
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
    with open(stats_path, 'w', encoding='utf-8') as f:
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
        if exp_type in data[user]:  # 保实验类型存在
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

def fit_and_evaluate_with_user_cv(data, degrees, n_splits=5):
    """使用基于用户的交叉验证进行评估"""
    users = list(data.keys())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {degree: {
        'train_data': [],
        'test_data': [],
        'train_errors': [],
        'test_errors': [],
        'train_r2': [],
        'test_r2': [],
        'models': []
    } for degree in degrees}
    
    # 进行交叉验证
    for fold, (train_idx, test_idx) in enumerate(kf.split(users)):
        # 分割训练集和测试集用户
        train_users = [users[i] for i in train_idx]
        test_users = [users[i] for i in test_idx]
        
        # 收集训练集和测试集数据
        train_measured, train_target = [], []
        test_measured, test_target = [], []
        
        # 收集训练集数据
        for user in train_users:
            user_measured, user_target = collect_user_data(data, user)
            train_measured.extend(user_measured)
            train_target.extend(user_target)
        
        # 收集测试集数据
        for user in test_users:
            user_measured, user_target = collect_user_data(data, user)
            test_measured.extend(user_measured)
            test_target.extend(user_target)
        
        # 对每个多项式阶数进行评估
        for degree in degrees:
            # 拟合模型
            calib_func, stats = fit_calibration_curve(train_measured, train_target, degree)
            
            # 预测
            train_pred = calib_func(np.array(train_measured))
            test_pred = calib_func(np.array(test_measured))
            
            # 计算误差和R²
            train_error = np.mean(np.abs(train_pred - np.array(train_target)))
            test_error = np.mean(np.abs(test_pred - np.array(test_target)))
            train_r2 = r2_score(train_target, train_pred)
            test_r2 = r2_score(test_target, test_pred)
            
            # 存储结果
            cv_results[degree]['train_data'].append((train_measured, train_target, train_pred))
            cv_results[degree]['test_data'].append((test_measured, test_target, test_pred))
            cv_results[degree]['train_errors'].append(train_error)
            cv_results[degree]['test_errors'].append(test_error)
            cv_results[degree]['train_r2'].append(train_r2)
            cv_results[degree]['test_r2'].append(test_r2)
            cv_results[degree]['models'].append((calib_func, stats))
        
    return cv_results

def plot_cv_results(cv_results, degrees, save_dir):
    """绘制交叉验证结果"""
    for fold in range(len(cv_results[degrees[0]]['train_data'])):
        plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=plt.gcf())
        
        # 1-4: 不同阶数的拟合效果图
        for i, degree in enumerate(degrees, 1):
            plt.subplot(gs[0, i-1])
            
            # 获取当前折的数据
            train_measured, train_target, train_pred = cv_results[degree]['train_data'][fold]
            test_measured, test_target, test_pred = cv_results[degree]['test_data'][fold]
            
            # 绘制训练集和测试集数据
            plt.scatter(train_measured, train_target, alpha=0.5, label='Train', color='blue')
            plt.scatter(train_measured, train_pred, alpha=0.5, label='Train Pred', color='lightblue')
            plt.scatter(test_measured, test_target, alpha=0.5, label='Test', color='red')
            plt.scatter(test_measured, test_pred, alpha=0.5, label='Test Pred', color='lightcoral')
            
            # 绘制拟合线
            calib_func, stats = cv_results[degree]['models'][fold]
            x_fit = np.linspace(min(train_measured + test_measured), 
                              max(train_measured + test_measured), 100)
            y_fit = calib_func(x_fit)
            plt.plot(x_fit, y_fit, 'g-', label='Calibration Function')
            
            plt.title(f'Degree {degree} Polynomial (Fold {fold+1})\n' + 
                     f'Train R² = {cv_results[degree]["train_r2"][fold]:.3f}\n' +
                     f'Test R² = {cv_results[degree]["test_r2"][fold]:.3f}')
            plt.xlabel('Measured Angle')
            plt.ylabel('Target/Calibrated Angle')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
        
        # 5-8: 每个阶数的误差箱线图对比
        for i, degree in enumerate(degrees):
            plt.subplot(gs[1, i])
            error_data = []
            labels = []
            
            # 获取数据
            train_measured, train_target, train_pred = cv_results[degree]['train_data'][fold]
            test_measured, test_target, test_pred = cv_results[degree]['test_data'][fold]
            
            # 原始误差
            original_error = np.abs(np.array(train_measured) - np.array(train_target))
            error_data.append(original_error)
            labels.append('Original')
            
            # 训练集误差
            train_error = np.abs(np.array(train_pred) - np.array(train_target))
            error_data.append(train_error)
            labels.append('Train')
            
            # 测试集误差
            test_error = np.abs(np.array(test_pred) - np.array(test_target))
            error_data.append(test_error)
            labels.append('Test')
            
            # 绘制箱线图
            plt.boxplot(error_data, labels=labels)
            plt.xticks(rotation=45)
            plt.ylabel('Absolute Error (degrees)')
            plt.title(f'Degree {degree} Error Distribution\n(Fold {fold+1})')
            plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'cv_fold_{fold+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_error_distribution_comparison(results, save_dir):
    """绘制校准前后误差分布的对比图"""
    # 添加调试信息
    print("Results keys:", results.keys())
    
    # 打印数据结构（避免 JSON 序列化）
    for key in results.keys():
        print(f"\nKey {key}:")
        if isinstance(results[key], dict):
            for subkey in results[key]:
                print(f"  {subkey}: {type(results[key][subkey])}")
        else:
            print(f"Type: {type(results[key])}")
    
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # 获取原始误差和校准后误差
    error_data = {}
    
    # 假设每个 key (1,2,3,4) 包含对应阶数的结果
    for degree in [1, 2, 3, 4]:
        if degree in results:
            if 'stats' in results[degree]:
                if degree == 1:  # 只在第一次时获取原始误差
                    error_data['Original'] = results[degree]['original_error']
                error_data[f'Degree {degree}'] = results[degree]['calibrated_error']
    
    # 1. 核密度估计图
    plt.subplot(gs[0, 0])
    for name, data in error_data.items():
        sns.kdeplot(data=data, label=name, alpha=0.5)
    
    plt.xlabel('Absolute Error (degrees)')
    plt.ylabel('Density')
    plt.title('Error Distribution (KDE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 箱线图
    plt.subplot(gs[0, 1])
    sns.boxplot(data=pd.DataFrame(error_data))
    plt.xticks(rotation=45)
    plt.ylabel('Absolute Error (degrees)')
    plt.title('Error Distribution (Box Plot)')
    plt.grid(True, alpha=0.3)
    
    # 3. 累积分布函数
    plt.subplot(gs[1, 0])
    for name, data in error_data.items():
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cumulative, label=name, alpha=0.7)
    
    plt.xlabel('Absolute Error (degrees)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 统计信息表格
    plt.subplot(gs[1, 1])
    plt.axis('off')
    stats_text = "Error Distribution Statistics:\n\n"
    
    for name, data in error_data.items():
        stats_text += f"{name}:\n"
        stats_text += f"Mean ± SD: {np.mean(data):.2f}° ± {np.std(data):.2f}°\n"
        stats_text += f"Median (IQR): {np.median(data):.2f}° "
        stats_text += f"({np.percentile(data, 25):.2f}° - {np.percentile(data, 75):.2f}°)\n\n"
    
    plt.text(0, 1, stats_text, fontsize=10, verticalalignment='top', 
             fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_calibration_stats(data, degrees):
    """计算校准前后的误差统计"""
    # 收集所有数据
    measured = []
    target = []
    
    # 从数据中收集测量值和目标值
    for user in data:
        for exp_type in ['1', '2']:
            if exp_type in data[user]:
                user_data = data[user][exp_type]
                for trial in user_data:
                    for platform in user_data[trial]:
                        if '_' not in platform:
                            measured.append(np.mean(user_data[trial][platform]['data']))
                            target.append(float(platform))
    
    measured = np.array(measured)
    target = np.array(target)
    
    # 计算原始误差
    original_error = np.abs(measured - target)
    
    # 为每个多项式阶数计算校准后的误差
    calibration_stats = {
        'original': {
            'mean': np.mean(original_error),
            'std': np.std(original_error),
            'median': np.median(original_error),
            'q1': np.percentile(original_error, 25),
            'q3': np.percentile(original_error, 75),
            'errors': original_error
        }
    }
    
    for degree in degrees:
        # 拟合校准曲线
        calib_func, _ = fit_calibration_curve(measured, target, degree)
        
        # 计算校准后的值和误差
        calibrated = calib_func(measured)
        calibrated_error = np.abs(calibrated - target)
        
        calibration_stats[f'degree_{degree}'] = {
            'mean': np.mean(calibrated_error),
            'std': np.std(calibrated_error),
            'median': np.median(calibrated_error),
            'q1': np.percentile(calibrated_error, 25),
            'q3': np.percentile(calibrated_error, 75),
            'errors': calibrated_error
        }
    
    # 生成报告
    report = "Calibration Results:\n\n"
    
    # 添加原始误差统计
    report += "Original Error:\n"
    report += f"Mean ± SD: {calibration_stats['original']['mean']:.2f}° ± {calibration_stats['original']['std']:.2f}°\n"
    report += f"Median (IQR): {calibration_stats['original']['median']:.2f}° "
    report += f"({calibration_stats['original']['q1']:.2f}° - {calibration_stats['original']['q3']:.2f}°)\n\n"
    
    # 添加每个阶数的校准结果
    for degree in degrees:
        stats = calibration_stats[f'degree_{degree}']
        report += f"Degree {degree} Polynomial:\n"
        report += f"Mean ± SD: {stats['mean']:.2f}° ± {stats['std']:.2f}°\n"
        report += f"Median (IQR): {stats['median']:.2f}° "
        report += f"({stats['q1']:.2f}° - {stats['q3']:.2f}°)\n\n"
    
    return calibration_stats, report

def calculate_calibration_stats_with_split(data, degrees, test_ratio=0.2):
    """计算训练集和测试集的校准统计"""
    # 收集所有数据
    measured = []
    target = []
    
    # 从数据中收集测量值和目标值
    for user in data:
        for exp_type in ['1', '2']:
            if exp_type in data[user]:
                user_data = data[user][exp_type]
                for trial in user_data:
                    for platform in user_data[trial]:
                        if '_' not in platform:
                            measured.append(np.mean(user_data[trial][platform]['data']))
                            target.append(float(platform))
    
    measured = np.array(measured)
    target = np.array(target)
    
    # 随机分割数据为训练集和测试集
    indices = np.random.permutation(len(measured))
    test_size = int(len(measured) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = measured[train_indices], measured[test_indices]
    y_train, y_test = target[train_indices], target[test_indices]
    
    # 计算原始误差
    train_original_error = np.abs(X_train - y_train)
    test_original_error = np.abs(X_test - y_test)
    
    # 存储统计结果
    calibration_stats = {
        'train': {
            'original': {
                'mean': np.mean(train_original_error),
                'std': np.std(train_original_error),
                'median': np.median(train_original_error),
                'q1': np.percentile(train_original_error, 25),
                'q3': np.percentile(train_original_error, 75),
                'errors': train_original_error
            }
        },
        'test': {
            'original': {
                'mean': np.mean(test_original_error),
                'std': np.std(test_original_error),
                'median': np.median(test_original_error),
                'q1': np.percentile(test_original_error, 25),
                'q3': np.percentile(test_original_error, 75),
                'errors': test_original_error
            }
        }
    }
    
    # 为每个多项式阶数计算校准后的误差
    for degree in degrees:
        # 拟合校准曲线（仅使用训练集）
        calib_func, _ = fit_calibration_curve(X_train, y_train, degree)
        
        # 计算训练集的校准误差
        train_calibrated = calib_func(X_train)
        train_calibrated_error = np.abs(train_calibrated - y_train)
        
        # 计算测试集的校准误差
        test_calibrated = calib_func(X_test)
        test_calibrated_error = np.abs(test_calibrated - y_test)
        
        # 存储训练集统计
        calibration_stats['train'][f'degree_{degree}'] = {
            'mean': np.mean(train_calibrated_error),
            'std': np.std(train_calibrated_error),
            'median': np.median(train_calibrated_error),
            'q1': np.percentile(train_calibrated_error, 25),
            'q3': np.percentile(train_calibrated_error, 75),
            'errors': train_calibrated_error
        }
        
        # 存储测试集统计
        calibration_stats['test'][f'degree_{degree}'] = {
            'mean': np.mean(test_calibrated_error),
            'std': np.std(test_calibrated_error),
            'median': np.median(test_calibrated_error),
            'q1': np.percentile(test_calibrated_error, 25),
            'q3': np.percentile(test_calibrated_error, 75),
            'errors': test_calibrated_error
        }
    
    # 生成报告
    report = "Calibration Results (Train-Test Split):\n\n"
    
    # 训练集报告
    report += "Training Set:\n" + "="*50 + "\n"
    report += "Original Error:\n"
    stats = calibration_stats['train']['original']
    report += f"Mean ± SD: {stats['mean']:.2f}° ± {stats['std']:.2f}°\n"
    report += f"Median (IQR): {stats['median']:.2f}° ({stats['q1']:.2f}° - {stats['q3']:.2f}°)\n\n"
    
    for degree in degrees:
        stats = calibration_stats['train'][f'degree_{degree}']
        report += f"Degree {degree} Polynomial:\n"
        report += f"Mean ± SD: {stats['mean']:.2f}° ± {stats['std']:.2f}°\n"
        report += f"Median (IQR): {stats['median']:.2f}° ({stats['q1']:.2f}° - {stats['q3']:.2f}°)\n\n"
    
    # 测试集报告
    report += "\nTest Set:\n" + "="*50 + "\n"
    report += "Original Error:\n"
    stats = calibration_stats['test']['original']
    report += f"Mean ± SD: {stats['mean']:.2f}° ± {stats['std']:.2f}°\n"
    report += f"Median (IQR): {stats['median']:.2f}° ({stats['q1']:.2f}° - {stats['q3']:.2f}°)\n\n"
    
    for degree in degrees:
        stats = calibration_stats['test'][f'degree_{degree}']
        report += f"Degree {degree} Polynomial:\n"
        report += f"Mean ± SD: {stats['mean']:.2f}° ± {stats['std']:.2f}°\n"
        report += f"Median (IQR): {stats['median']:.2f}° ({stats['q1']:.2f}° - {stats['q3']:.2f}°)\n\n"
    
    return calibration_stats, report

def plot_train_test_error_distribution(stats, save_dir):
    """绘制训练集和测试集的误差分布对比图"""
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # 1. 训练集核密度估计图
    plt.subplot(gs[0, 0])
    sns.kdeplot(data=stats['train']['original']['errors'], 
                label='Original', color='red', alpha=0.5)
    for degree in [1, 2, 3, 4]:
        sns.kdeplot(data=stats['train'][f'degree_{degree}']['errors'], 
                   label=f'Degree {degree}', alpha=0.5)
    
    plt.xlabel('Absolute Error (degrees)')
    plt.ylabel('Density')
    plt.title('Training Set Error Distribution (KDE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 测试集核密度估计图
    plt.subplot(gs[0, 1])
    sns.kdeplot(data=stats['test']['original']['errors'], 
                label='Original', color='red', alpha=0.5)
    for degree in [1, 2, 3, 4]:
        sns.kdeplot(data=stats['test'][f'degree_{degree}']['errors'], 
                   label=f'Degree {degree}', alpha=0.5)
    
    plt.xlabel('Absolute Error (degrees)')
    plt.ylabel('Density')
    plt.title('Test Set Error Distribution (KDE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 训练集和测试集箱线图
    plt.subplot(gs[1, :])
    error_data = {
        'Train Original': stats['train']['original']['errors'],
        'Test Original': stats['test']['original']['errors']
    }
    for degree in [1, 2, 3, 4]:
        error_data[f'Train Degree {degree}'] = stats['train'][f'degree_{degree}']['errors']
        error_data[f'Test Degree {degree}'] = stats['test'][f'degree_{degree}']['errors']
    
    sns.boxplot(data=pd.DataFrame(error_data))
    plt.xticks(rotation=45)
    plt.ylabel('Absolute Error (degrees)')
    plt.title('Error Distribution Comparison (Train vs Test)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_test_error_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

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
    print("分析整模型在个人数据上的效果...")
    overall_model_results = analyze_overall_calibration_effect(
        data, degrees, save_dir
    )
    
    # 创建交叉验证结果目录
    cv_dir = os.path.join(save_dir, 'cross_validation')
    ensure_dir(cv_dir)
    
    # 进行基于用户的交叉验证
    print("进行基于用户的交叉验证...")
    cv_results = fit_and_evaluate_with_user_cv(data, degrees)
    
    # 绘制交叉验证结果
    plot_cv_results(cv_results, degrees, cv_dir)
    
    # 生成交叉验证报告
    cv_report = "Cross-Validation Results:\n\n"
    for degree in degrees:
        cv_report += f"\nDegree {degree} Polynomial:\n"
        cv_report += f"Train Error: {np.mean(cv_results[degree]['train_errors']):.2f}° ± {np.std(cv_results[degree]['train_errors']):.2f}°\n"
        cv_report += f"Test Error: {np.mean(cv_results[degree]['test_errors']):.2f}° ± {np.std(cv_results[degree]['test_errors']):.2f}°\n"
        cv_report += f"Train R²: {np.mean(cv_results[degree]['train_r2']):.3f} ± {np.std(cv_results[degree]['train_r2']):.3f}\n"
        cv_report += f"Test R²: {np.mean(cv_results[degree]['test_r2']):.3f} ± {np.std(cv_results[degree]['test_r2']):.3f}\n"
    
    with open(os.path.join(cv_dir, 'cv_report.txt'), 'w', encoding='utf-8') as f:
        f.write(cv_report)
    
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
    
    # 计算校准统计
    print("计算校准统计...")
    calibration_stats, calibration_report = calculate_calibration_stats(data, degrees)
    
    # 保存报告
    with open(os.path.join(save_dir, 'calibration_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(calibration_report)
    
    # 计算训练集和测试集的校准统计
    print("计算训练集和测试集的校准统计...")
    split_stats, split_report = calculate_calibration_stats_with_split(data, degrees)
    
    # 保存报告
    with open(os.path.join(save_dir, 'train_test_calibration_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(split_report)
    
    # # 绘制误差分布对比图
    # print("绘制误差分布对比图...")
    # plot_error_distribution_comparison(calibration_stats, save_dir)

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