import os
import re
import json
import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, 
    QFileDialog, QComboBox, QMessageBox, QGroupBox, QCheckBox,QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QLocale
from PyQt5.QtGui import QIcon

# 导入主流程窗口和数据处理器（目前仅为结构占位，需要您实现完整内容）
# 注意：在实际项目中，需要确保这些文件和类已定义
from main_pipeline_window import MainPipelineWindow
from minian_processor import MinianProcessor # 假设存在

class StartupWindow(QWidget):
    """
    Minian UI 启动窗口，处理文件夹选择、流程选择和参数配置文件的初始化。
    """
    
    # 窗口标题和默认配置路径
    WINDOW_TITLE = "Minian UI - 启动配置"
    DEFAULT_CONFIG_NAME = "default_config.json"
    # 配置默认库地址
    DEFAULT_VIDEO_FOLDER = "D:\Desktop\ZJU\SRTP\demo"
    
    def __init__(self):
        super().__init__()
        # 强制使用中文语言环境以确保弹窗和UI元素显示中文
        QLocale.setDefault(QLocale(QLocale.Chinese, QLocale.China))
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setGeometry(300, 300, 700, 400)
        
        self.dpath = "D:\Desktop\ZJU\SRTP\demo"  # 视频文件夹路径
        self.pipeline_window = None # 用于存储主流程窗口实例
        
        self.init_ui()
        self.load_default_folder()  # 直接加载默认文件夹

    def init_ui(self):
        """初始化用户界面布局和控件。"""
        main_layout = QVBoxLayout(self)
        
        # 1. 文件夹选择组
        folder_group = QGroupBox("1. 视频文件夹设置")
        folder_layout = QHBoxLayout()
        self.path_label = QLineEdit("请选择视频文件夹...")
        self.path_label.setReadOnly(True)
        self.path_label.setStyleSheet("color: blue;")
        self.select_btn = QPushButton("选择文件夹")
        self.select_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.path_label)
        folder_layout.addWidget(self.select_btn)
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # 2. 正则表达式和流程选择组
        config_group = QGroupBox("2. 流程配置与启动")
        config_layout = QVBoxLayout()
        
        # 正则表达式
        regex_layout = QHBoxLayout()
        regex_layout.addWidget(QLabel("视频文件名正则:"))
        self.regex_input = QLineEdit(r"msCam[0-9]+\.avi$")
        regex_layout.addWidget(self.regex_input)
        config_layout.addLayout(regex_layout)
        
        # 流程选择
        process_layout = QHBoxLayout()
        process_layout.addWidget(QLabel("选择处理流程:"))
        self.process_combo = QComboBox()
        self.process_combo.addItems(["Minian 标准流程", "自定义流程 (未设计)"])
        process_layout.addWidget(self.process_combo)
        config_layout.addLayout(process_layout)

        # 配置文件操作
        config_options_group = QGroupBox("参数配置库操作")
        config_options_layout = QHBoxLayout()
        
        self.new_config_radio = QCheckBox("创建新配置 (使用默认参数)")
        self.new_config_radio.setChecked(True)
        self.import_config_radio = QCheckBox("导入现有配置")
        
        # 确保互斥
        self.new_config_radio.toggled.connect(lambda state: self.import_config_radio.setChecked(not state))
        self.import_config_radio.toggled.connect(lambda state: self.new_config_radio.setChecked(not state))

        self.import_config_path = QLineEdit("导入配置路径...")
        self.import_config_path.setReadOnly(True)
        self.import_config_btn = QPushButton("选择配置")
        self.import_config_btn.clicked.connect(self.select_config_file)
        
        import_path_layout = QHBoxLayout()
        import_path_layout.addWidget(self.import_config_path)
        import_path_layout.addWidget(self.import_config_btn)
        
        config_options_layout.addWidget(self.new_config_radio)
        config_options_layout.addWidget(self.import_config_radio)
        
        config_options_group.setLayout(config_options_layout)
        config_layout.addWidget(config_options_group)
        config_layout.addLayout(import_path_layout)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # 3. 开始按键
        self.start_btn = QPushButton("开始主流程")
        self.start_btn.setEnabled(False) # 初始禁用，直到选择文件夹
        self.start_btn.clicked.connect(self.start_main_pipeline)
        main_layout.addWidget(self.start_btn)
        
        main_layout.addStretch(1)

    def load_default_folder(self):
        """加载默认的视频文件夹路径。"""
        if os.path.isdir(self.dpath):
            self.path_label.setText(self.dpath)
            self.start_btn.setEnabled(True)
            self.generate_regex_from_folder(self.dpath)
            print(f"终端交互: 使用默认视频文件夹路径: {self.dpath}")
        else:
            QMessageBox.warning(self, "警告", f"默认视频文件夹路径无效: {self.dpath}\n请手动选择一个有效的文件夹。")
            self.dpath = ""
            self.path_label.setText("请选择视频文件夹...")
            self.start_btn.setEnabled(False)
            print(f"终端交互: 默认视频文件夹路径无效: {self.dpath}")
        
    def select_folder(self):
        """打开文件对话框选择视频文件夹。"""
        print("终端交互: 正在打开文件夹选择对话框...")
        folder = QFileDialog.getExistingDirectory(self, "选择包含视频文件的文件夹", self.dpath)
        
        if folder:
            self.dpath = folder
            self.path_label.setText(self.dpath)
            print(f"终端交互: 选定文件夹: {self.dpath}")
            self.start_btn.setEnabled(True)
            self.generate_regex_from_folder(folder)
        else:
            print("终端交互: 取消选择文件夹。")
        # 如果用户取消选择文件夹，保持当前路径不变

    def generate_regex_from_folder(self, folder_path):
        """自动检查文件夹内的视频文件，并尝试生成合适的正则表达式。"""
        video_extensions = ['.avi', '.tif', '.tiff', '.mp4']
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        video_files = [f for f in files if any(f.lower().endswith(ext) for ext in video_extensions)]
        
        if video_files:
            # 找到第一个视频文件，尝试猜默认 Minian 模式
            first_file = video_files[0]
            print(f"终端交互: 文件夹中找到第一个视频文件: {first_file}")
            
            # 尝试 Minian 的默认模式 'msCam[0-9]+\.avi$'
            if re.match(r"msCam[0-9]+\.avi$", first_file):
                default_pattern = r"msCam[0-9]+\.avi$"
            # 尝试 TIF 序列模式
            elif first_file.lower().endswith('.tif') or first_file.lower().endswith('.tiff'):
                default_pattern = r".*\.tif[f]?$"
            else:
                # 通用模式
                default_pattern = r".*\.(avi|tif|tiff|mp4)$"
            
            self.regex_input.setText(default_pattern)
            print(f"终端交互: 自动生成正则表达式: {default_pattern}")
        else:
            QMessageBox.warning(self, "警告", "文件夹中未找到常见的视频文件 (*.avi, *.tif 等)。请手动输入正确的正则表达式。")
            print("终端交互: 未检测到视频文件，保持默认正则表达式。")


    def select_config_file(self):
        """选择要导入的参数配置文件（JSON格式）。"""
        if not self.import_config_radio.isChecked():
            return
            
        print("终端交互: 正在打开配置文件选择对话框...")
        config_file, _ = QFileDialog.getOpenFileName(
            self, 
            "选择参数配置文件", 
            self.dpath if self.dpath else os.getcwd(), 
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        
        if config_file:
            self.import_config_path.setText(config_file)
            print(f"终端交互: 选定配置文件: {config_file}")

    
    def start_main_pipeline(self):
        """
        根据用户选择初始化参数配置，并启动主流程窗口。
        """
        video_folder = self.dpath
        regex_pattern = self.regex_input.text()
        
        # 1. 检查文件夹路径和正则
        if not os.path.isdir(video_folder):
            QMessageBox.critical(self, "错误", "请选择有效的视频文件夹路径。")
            return
            
        # 2. 确定配置文件的路径（仓库路径）
        config_base_dir = os.path.join(video_folder, "minian_repo")
        if not os.path.exists(config_base_dir):
            os.makedirs(config_base_dir)
            
        try:
            intermediate_path = os.path.join(video_folder, "minian_intermediate")
            
            # 1. 设置环境变量
            os.environ["MINIAN_INTERMEDIATE"] = intermediate_path
            
            # 2. 确保文件夹存在
            os.makedirs(intermediate_path, exist_ok=True)
            print(f"终端交互: MINIAN_INTERMEDIATE 已设置为: {intermediate_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法创建或设置中间数据路径: {e}")
            print(f"错误: 无法创建或设置中间数据路径: {e}")
            return
            
        config_path = os.path.join(config_base_dir, "config.json")
        
        # 3. 处理配置文件初始化逻辑
        try:
            if self.new_config_radio.isChecked():
                # 复制 default_config.json 到仓库
                default_config_src = self.DEFAULT_CONFIG_NAME
                if not os.path.exists(default_config_src):
                    # 如果默认配置文件不存在，则创建空的
                    print(f"警告: {self.DEFAULT_CONFIG_NAME} 未找到，创建空的配置文件。")
                    default_content = {}
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(default_content, f, indent=4)
                else:
                    # 复制默认配置
                    with open(default_config_src, 'r', encoding='utf-8') as src_f:
                        default_content = json.load(src_f)
                    with open(config_path, 'w', encoding='utf-8') as dst_f:
                        json.dump(default_content, dst_f, indent=4)
                
                print(f"终端交互: 已使用默认参数创建新的配置库文件: {config_path}")
                QMessageBox.information(self, "信息", f"已创建新配置库文件:\n{config_path}\n请在主界面继续运行。")
                
            elif self.import_config_radio.isChecked():
                # 复制导入的配置到仓库
                import_src = self.import_config_path.text()
                if not os.path.exists(import_src):
                    QMessageBox.critical(self, "错误", "导入的配置文件路径无效。")
                    return
                
                # 复制文件
                with open(import_src, 'r', encoding='utf-8') as src_f:
                    imported_content = json.load(src_f)
                with open(config_path, 'w', encoding='utf-8') as dst_f:
                    json.dump(imported_content, dst_f, indent=4)
                    
                print(f"终端交互: 已复制外部配置到配置库文件: {config_path}")
                QMessageBox.information(self, "信息", f"已导入外部配置并创建配置库文件:\n{config_path}")
                
        except json.JSONDecodeError:
            QMessageBox.critical(self, "错误", "配置库文件（或导入文件）的格式不正确（非标准 JSON）。")
            return
        except Exception as e:
            QMessageBox.critical(self, "错误", f"配置文件操作失败: {e}")
            print(f"错误: 配置文件操作失败: {e}")
            return
        
        # 4. 实例化 MinianProcessor
        try:
            processor = MinianProcessor(video_folder, config_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化 Minian 处理器失败，可能与 Dask 或核心库文件有关: {e}")
            print(f"错误: 初始化 Minian 处理器失败: {e}")
            return
            
        # 5. 启动主流程窗口
        try:
            self.pipeline_window = MainPipelineWindow(
                processor=processor, 
                pipeline_mode=self.process_combo.currentText(),
                regex_pattern=regex_pattern
            )
            self.pipeline_window.show()
            self.hide() # 隐藏启动窗口
            print("终端交互: 成功启动主流程窗口。")
        except Exception as e:
            QMessageBox.critical(self.pipeline_window, "错误", f"启动主流程窗口失败: {e}")
            print(f"错误: 启动主流程窗口失败: {e}")
            # 重新显示启动窗口
            self.show()