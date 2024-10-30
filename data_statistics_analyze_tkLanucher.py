import tkinter as tk
from tkinter import ttk
import os
import importlib
import sys
from rich.console import Console
console = Console()

class AnalyzerLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Launcher")
        
        # 设置窗口大小和位置
        window_width = 400
        window_height = 300
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 分析方法选择
        ttk.Label(main_frame, text="Select Analysis Method:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar()
        self.method_combo = ttk.Combobox(main_frame, textvariable=self.method_var)
        self.method_combo['values'] = [
            "Personal Platform Data Comparison",
            # 在这里添加更多分析方法
        ]
        self.method_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        self.method_combo.set("Personal Platform Data Comparison")
        
        # 运行按钮
        ttk.Button(main_frame, text="Run Analysis", command=self.run_analysis).grid(row=2, column=0, pady=20)
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=3, column=0, pady=5)
        
    def run_analysis(self):
        """运行选中的分析方法"""
        method = self.method_var.get()
        self.status_var.set(f"Running {method}...")
        self.root.update()
        
        try:
            if method == "Personal Platform Data Comparison":
                # 导入并运行个人平台数据对比分析
                module = importlib.import_module("data_personal_platformData_comparsion")
                console.print("[green]Starting Personal Platform Data Comparison analysis...[/green]")
                if hasattr(module, '__main__'):
                    module.__main__()
                console.print("[green]Analysis completed successfully![/green]")
            # 在这里添加更多方法的处理
            
            self.status_var.set("Analysis completed successfully!")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            console.print(f"[red]Error occurred: {str(e)}[/red]")

def main():
    root = tk.Tk()
    app = AnalyzerLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()