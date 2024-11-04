import streamlit as st
import os
import json
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import numpy as np

def load_data(base_path):
    """加载校准结果数据"""
    result_path = os.path.join(base_path, "result")
    with open(os.path.join(result_path, "calibration_errors.json"), 'r') as f:
        errors = json.load(f)
    return errors

def create_sidebar():
    """创建侧边栏"""
    st.sidebar.title("校准结果分析")
    st.sidebar.markdown("---")
    return st.sidebar.radio(
        "选择查看内容",
        ["总体概览", "个人误差校准结果", "平台值细节"]
    )

def show_overview(base_path):
    """显示总体概览页面"""
    st.title("VOG校准结果总览")
    st.markdown("---")
    
    col1, _ = st.columns([2, 1])  # 创建2:1的列布局
    with col1:
        dist_img_path = os.path.join(base_path, "fig", "calibration_analysis",
                                    "error_distribution_analysis.png")
        if os.path.exists(dist_img_path):
            dist_img = Image.open(dist_img_path)
            st.image(dist_img, width=800)  # 固定宽度
        else:
            st.warning("未找到误差分布图片")

def show_personal_analysis(base_path):
    """显示个人分析页面"""
    st.title("个人校准效果分析")
    st.markdown("---")
    
    # 获取所有用户文件夹
    users_path = os.path.join(base_path, "fig", "calibration_analysis")
    if os.path.exists(users_path):
        users = [d for d in os.listdir(users_path)
                if os.path.isdir(os.path.join(users_path, d))]
        
        if users:
            # 用户选择
            selected_user = st.selectbox("选择用户", users)
            
            user_path = os.path.join(users_path, selected_user)
            
            # 展示用户的平均效果图
            st.subheader("校准平均效果")
            avg_img_path = os.path.join(user_path, f"{selected_user}_average_comparison.png")
            if os.path.exists(avg_img_path):
                avg_img = Image.open(avg_img_path)
                st.image(avg_img, width=700)  # 固定宽度
            else:
                st.warning("未找到平均效果图片")
            
            # 展示各次实验的结果
            st.subheader("各次实验校准效果")
            exp_files = [f for f in os.listdir(user_path) if "trial" in f]
            
            if exp_files:
                # 创建标签页
                tabs = st.tabs([f"实验 {i+1}" for i in range(len(exp_files))])
                for tab, file in zip(tabs, sorted(exp_files)):
                    with tab:
                        img_path = os.path.join(user_path, file)
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            st.image(img, width=700)  # 固定宽度
                        else:
                            st.warning(f"未找到实验图片: {file}")
            else:
                st.info("未找到实验结果图片")
        else:
            st.warning("未找到用户数据")
    else:
        st.error("未找到分析结果目录")

def show_error_distribution(errors):
    """显示误差分布页面"""
    st.title("校准误差分布分析")
    st.markdown("---")
    
    # 准备数据
    orig_errors = []
    calib_errors = []
    
    for person in errors['original']:
        for exp_type in errors['original'][person]:
            orig_errors.extend(errors['original'][person][exp_type].values())
            calib_errors.extend(errors['calibrated'][person][exp_type].values())
    
    # 创建箱型图
    fig = go.Figure()
    fig.add_trace(go.Box(y=orig_errors, name="原始误差", marker_color='red'))
    fig.add_trace(go.Box(y=calib_errors, name="校准差", marker_color='blue'))
    
    fig.update_layout(
        title="误差分布对比",
        yaxis_title="误差值",
        showlegend=True,
        boxmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示统计信息
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始误差统计")
        st.write(f"平均值: {np.mean(orig_errors):.2f}")
        st.write(f"中位数: {np.median(orig_errors):.2f}")
        st.write(f"标准差: {np.std(orig_errors):.2f}")
    
    with col2:
        st.subheader("校准后误差统计")
        st.write(f"平均值: {np.mean(calib_errors):.2f}")
        st.write(f"中位数: {np.median(calib_errors):.2f}")
        st.write(f"标准差: {np.std(calib_errors):.2f}")

def show_platform_comparisons(base_path):
    """显示平台对比分析页面"""
    st.title("平台对比分析")
    st.markdown("---")
    
    # 获取所有用户
    users_path = os.path.join(base_path, "fig", "calibration_analysis")
    users = []
    if os.path.exists(users_path):
        users = [d for d in os.listdir(users_path)
                if os.path.isdir(os.path.join(users_path, d))]
    
    # 平台值列表
    platform_values = ["0_1", "0_2", "5", "10", "15", "25", "-5", "-10", "-15", "-25"]
    
    # 创建两列布局用于选择
    col1, col2 = st.columns(2)
    with col1:
        selected_user = st.selectbox("选择用户", users if users else ["未找到用户"])
    with col2:
        selected_platform = st.selectbox("选择平台值", platform_values)
    
    if users:
        tab1, tab2, tab3 = st.tabs([
            "个人平台值对比", 
            "个人相对步长对比", 
            "总体平台值对比"
        ])
        
        # 个人平台对比
        with tab1:
            st.subheader(f"{selected_user}的平台对比分析")
            personal_path = os.path.join(base_path, "fig", "personal_platform_comparison", selected_user)
            
            if os.path.exists(personal_path):
                # 首先显示全部平台对比图
                all_platforms_path = os.path.join(personal_path, "all_platforms_comparison.png")
                if os.path.exists(all_platforms_path):
                    st.markdown("### 全部平台对比")
                    img = Image.open(all_platforms_path)
                    st.image(img, width=700)
                    st.markdown("---")
                
                # 显示单个平台的对比图
                st.markdown(f"### 平台值 {selected_platform} 的对比")
                user_platform_images = [f for f in sorted(os.listdir(personal_path)) 
                                     if selected_platform in f and 
                                     f.endswith('.png')]
                
                if user_platform_images:
                    for img_file in user_platform_images:
                        img = Image.open(os.path.join(personal_path, img_file))
                        caption = img_file.replace(".png", "")
                        st.image(img, caption=caption, width=700)
                else:
                    st.info(f"未找到平台值 {selected_platform} 的对比图")
            else:
                st.warning(f"未找到用户 {selected_user} 的数据文件夹")
        
        # 个人相对对比
        with tab2:
            st.subheader(f"{selected_user}的相对对比分析")
            relative_base_path = os.path.join(base_path, "fig", "personal_platform_relative_comparison", selected_user)
            
            if os.path.exists(relative_base_path):
                # 显示柱状图对比
                bars_img_path = os.path.join(relative_base_path, "platform_steps_comparison_bars.png")
                if os.path.exists(bars_img_path):
                    st.markdown("### 平台步长对比")
                    img = Image.open(bars_img_path)
                    st.image(img, width=700)  # 固定宽度
                    st.markdown("---")
                
                # 显示累积对比图
                cumulative_img_path = os.path.join(relative_base_path, "platform_steps_comparison_with_cumulative.png")
                if os.path.exists(cumulative_img_path):
                    st.markdown("### 累积步长对比")
                    img = Image.open(cumulative_img_path)
                    st.image(img, width=700)  # 固定宽度
                else:
                    st.info("未找到累积对比图")
            else:
                st.warning(f"未找到用户 {selected_user} 的相对对比数据")
                
        # 总体平台值对比
        with tab3:
            st.subheader("总体平台值对比分析")
            multi_path = os.path.join(base_path, "fig", "multiperson_platform_comparison")
            
            if os.path.exists(multi_path):
                # 显示平均值对比图
                mean_img_path = os.path.join(multi_path, "platform_person_comparison_mean.png")
                if os.path.exists(mean_img_path):
                    st.markdown("### 平均值对比")
                    img = Image.open(mean_img_path)
                    st.image(img, width=700)
                    st.markdown("---")
                
                # 显示中位数对比图
                median_img_path = os.path.join(multi_path, "platform_person_comparison_median.png")
                if os.path.exists(median_img_path):
                    st.markdown("### 中位数对比")
                    img = Image.open(median_img_path)
                    st.image(img, width=700)
                else:
                    st.info("未找到中位数对比图")
            else:
                st.warning("未找到总体平台值对比数据")
    else:
        st.error("未找到用户数据")


def main():
    st.set_page_config(
        page_title="VOG校准结果分析",
        page_icon="📊",
        layout="wide"
    )
    
    # 设置基础路径并检查
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 检查必要的目录是否存在
    if not os.path.exists(os.path.join(base_path, "fig")):
        st.error("未找到fig目录")
        return
    
    if not os.path.exists(os.path.join(base_path, "result")):
        st.error("未找到result目录")
        return
    
    # 加载数据
    errors = load_data(base_path)
    
    # 创建侧边栏
    page = create_sidebar()
    
    # 根据选择显示不同页面
    if page == "总体概览":
        show_overview(base_path)
        show_error_distribution(errors)
    elif page == "个人误差校准结果":
        show_personal_analysis(base_path)
    elif page == "平台值细节":
        show_platform_comparisons(base_path)

if __name__ == "__main__":
    main()