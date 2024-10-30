import streamlit as st
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
import plistlib

def read_plist_data(filename):
    with open(filename, 'rb') as f:
        data = plistlib.load(f)
        
    data = np.array(data)
    
    return data

def get_platform_data(data, platform_ranges):
    platform_data = {}
    platforms = ["0_1", 5, 10, 15, 25, "0_2", -5, -10, -15, -25]
    
    for platform, (start_idx, end_idx) in zip(platforms, platform_ranges):
        interval_data = data[start_idx:end_idx].tolist()
        platform_data[str(platform)] = interval_data
    
    return platform_data

def plot_data_with_range_selector(data, filename, platform_idx, prev_end_idx=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='数据'))
    
    fig.update_layout(
        title=f"文件: {filename}",
        xaxis_title="数据点索引",
        yaxis_title="值",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{platform_idx}")
    
    # 修改session_state的初始值为0
    if f"start_{platform_idx}" not in st.session_state:
        st.session_state[f"start_{platform_idx}"] = 0  # 始终从0开始
    if f"end_{platform_idx}" not in st.session_state:
        st.session_state[f"end_{platform_idx}"] = 0
    
    # 创建两列布局用于输入范围
    col1, col2 = st.columns(2)
    with col1:
        start_idx = st.number_input("起始索引", 0, len(data)-1, 
                                  st.session_state[f"start_{platform_idx}"], 
                                  key=f"start_input_{platform_idx}",
                                  on_change=lambda: setattr(st.session_state, f"start_{platform_idx}", 
                                                         st.session_state[f"start_input_{platform_idx}"]))
    with col2:
        end_idx = st.number_input("结束索引", 0, len(data)-1, 
                                 st.session_state[f"end_{platform_idx}"], 
                                 key=f"end_input_{platform_idx}",
                                 on_change=lambda: setattr(st.session_state, f"end_{platform_idx}", 
                                                        st.session_state[f"end_input_{platform_idx}"]))
        
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
        platforms = ["0_1", 5, 10, 15, 25, "0_2", -5, -10, -15, -25]
        
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
            
            platforms = ["0_1", 5, 10, 15, 25, "0_2", -5, -10, -15, -25]
            platform_ranges = []
            prev_end_idx = None
            
            for platform_idx, platform in enumerate(platforms):
                st.subheader(f"选择平台 {platform}度 的范围")
                start_idx, end_idx = plot_data_with_range_selector(data, filename, platform_idx, prev_end_idx)
                platform_ranges.append((start_idx, end_idx))
                prev_end_idx = end_idx
            
            # 添加最终保存按钮
            if st.button("保存到JSON文件"):
                try:
                    result = {
                        "name": selected_subject.name,
                        "experiments": {
                            exp_num: {
                                trial_num: get_platform_data(data, platform_ranges)
                            }
                        }
                    }
                    
                    # 确保result目录存在
                    Path("result").mkdir(exist_ok=True)
                    # 创建一个以受试者名字命名的文件夹
                    subject_folder = Path("result") / selected_subject.name
                    subject_folder.mkdir(exist_ok=True)
                    # 生成文件名，包括实验编号和试验编号，防止重名
                    output_file = subject_folder / f"{exp_num}-{trial_num}_platforms.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=4)
                    st.success(f"已保存结果到 {output_file}")
                except Exception as e:
                    st.error(f"保存失败：{str(e)}")

if __name__ == "__main__":
    main()
