import streamlit as st
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path

def read_plist_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if '<real>' in line:
                value = float(line.split('>')[1].split('<')[0])
                data.append(value)
    return np.array(data)

def get_platform_data(data, platform_ranges):
    platform_data = {}
    platforms = [0, 5, 10, 15, 25, 0, -5, -10, -15, -25]
    
    for platform, (start_idx, end_idx) in zip(platforms, platform_ranges):
        interval_data = data[start_idx:end_idx].tolist()
        platform_data[str(platform)] = interval_data
    
    return platform_data

def plot_data_with_range_selector(data, filename, platform_idx):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='数据'))
    
    # 设置图表布局
    fig.update_layout(
        title=f"文件: {filename}",
        xaxis_title="数据点索引",
        yaxis_title="值",
        height=400
    )
    
    # 显示图表并返回用户选择的范围
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{platform_idx}")
    
    # 创建两列布局用于输入范围
    col1, col2 = st.columns(2)
    with col1:
        start_idx = st.number_input("起始索引", 0, len(data)-1, 0, 
                                  key=f"start_{platform_idx}")
    with col2:
        end_idx = st.number_input("结束索引", 0, len(data)-1, min(100, len(data)-1), 
                                 key=f"end_{platform_idx}")
        
    return start_idx, end_idx

def process_subject_data(subject_dir):
    subject_name = subject_dir.name
    result = {
        "name": subject_name,
        "experiments": {}
    }
    
    # 遍历所有实验文件
    for file_path in subject_dir.glob("*_lefteye.plist"):
        filename = file_path.stem
        exp_trial = filename.split('_')[0]
        exp_num = int(exp_trial[len(subject_name)])
        trial_num = int(exp_trial.split('-')[1])
        
        if exp_num not in result["experiments"]:
            result["experiments"][exp_num] = {}
            
        data = read_plist_data(file_path)
        
        # 使用Streamlit显示文件信息
        st.header(f"处理文件: {filename}")
        
        platform_ranges = []
        platforms = [0, 5, 10, 15, 25, 0, -5, -10, -15, -25]
        
        for platform_idx, platform in enumerate(platforms):
            st.subheader(f"选择平台 {platform}度 的范围")
            start_idx, end_idx = plot_data_with_range_selector(data, filename, platform_idx)
            platform_ranges.append((start_idx, end_idx))
            
            # 添加一个确认按钮
            if st.button(f"确认平台{platform}度的范围", key=f"confirm_{platform_idx}"):
                st.success(f"已保存平台{platform}度的范围: {start_idx} - {end_idx}")
        
        platform_data = get_platform_data(data, platform_ranges)
        result["experiments"][exp_num][trial_num] = platform_data
    
    return result

def main():
    st.title("平台数据范围选择工具")
    
    # 固定使用data目录
    data_dir = Path("data")
    if not data_dir.exists():
        st.error("data目录不存在！")
        return
        
    # 在侧边栏显示文件列表
    st.sidebar.header("文件选择")
    
    # 获取所有受试者文件夹
    subject_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    selected_subject = st.sidebar.selectbox(
        "选择受试者:", 
        options=subject_dirs,
        format_func=lambda x: x.name
    )
    
    if selected_subject:
        # 获取选中受试者文件夹下的所有plist文件
        plist_files = list(selected_subject.glob("*_lefteye.plist"))
        selected_file = st.sidebar.selectbox(
            "选择文件:",
            options=plist_files,
            format_func=lambda x: x.stem
        )
        
        if selected_file:
            # 处理选中的文件
            data = read_plist_data(selected_file)
            filename = selected_file.stem
            exp_trial = filename.split('_')[0]
            exp_num = int(exp_trial[len(selected_subject.name)])
            trial_num = int(exp_trial.split('-')[1])
            
            st.header(f"处理文件: {filename}")
            
            # 初始化或获取session state中的数据
            if 'platform_ranges' not in st.session_state:
                st.session_state.platform_ranges = []
            
            platforms = [0, 5, 10, 15, 25, 0, -5, -10, -15, -25]
            
            # 添加"保存所有范围"按钮
            save_all = st.button("保存所有范围")
            
            for platform_idx, platform in enumerate(platforms):
                st.subheader(f"选择平台 {platform}度 的范围")
                start_idx, end_idx = plot_data_with_range_selector(data, filename, platform_idx)
                
                # 如果点击了保存所有范围，或者单独确认了这个范围
                if (save_all or 
                    st.button(f"确认平台{platform}度的范围", key=f"confirm_{platform_idx}")):
                    if platform_idx >= len(st.session_state.platform_ranges):
                        st.session_state.platform_ranges.append((start_idx, end_idx))
                    else:
                        st.session_state.platform_ranges[platform_idx] = (start_idx, end_idx)
                    st.success(f"已保存平台{platform}度的范围: {start_idx} - {end_idx}")
            
            # 添加最终保存按钮
            if st.button("保存到JSON文件"):
                if len(st.session_state.platform_ranges) == len(platforms):
                    result = {
                        "name": selected_subject.name,
                        "experiments": {
                            exp_num: {
                                trial_num: get_platform_data(data, st.session_state.platform_ranges)
                            }
                        }
                    }
                    
                    output_file = f"{selected_subject.name}_platforms.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=4)
                    st.success(f"已保存结果到 {output_file}")
                    
                    # 清空已保存的范围，准备处理下一个文件
                    st.session_state.platform_ranges = []
                else:
                    st.warning("请先确认所有平台的范围！")

if __name__ == "__main__":
    main()