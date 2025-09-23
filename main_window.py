import os
import re
from typing import Optional, List, Tuple
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QFileDialog, QLineEdit, QPushButton, QComboBox,
    QFrame, QSpacerItem, QSizePolicy, QProgressBar, QMessageBox, QDesktopWidget
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QColor, QFont, QPainter, QImage
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QGraphicsOpacityEffect, QSlider
import xarray as xr

from .data_manager import DataManager
from .minian_pipeline import MinianPipeline
from .visualization_pyside import VideoPlayer, ComparisonVideoPlayer, CNMFVisualizer
from .widgets import ParamPanel, StepNavigation
from .config import STEP_NAMES
from natsort import natsorted
from .minian_core import utilities


def _longest_common_prefix(strs: list) -> str:
    """计算列表中所有字符串的最长公共前缀"""
    if not strs:
        return ""
    s1 = min(strs)
    s2 = max(strs)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1

def _longest_common_suffix(strs: list) -> str:
    """计算列表中所有字符串的最长公共后缀"""
    if not strs:
        return ""
    reversed_strs = [s[::-1] for s in strs]
    reversed_prefix = _longest_common_prefix(reversed_strs)
    return reversed_prefix[::-1]


class VideoLoader(QThread):
    """
    一个用于在单独线程中加载视频文件的工作类。
    """
    video_loaded = pyqtSignal(object)
    
    def __init__(self, movie_path: str, pattern: str):
        super().__init__()
        self.movie_path = movie_path
        self.pattern = pattern
    
    def run(self):
        try:
            varr = utilities.load_videos(self.movie_path, self.pattern)
            self.video_loaded.emit(varr)
        except Exception as e:
            # 捕获并打印出具体的异常信息
            print(f"视频加载失败，请检查文件格式或编解码器问题: {e}")
            self.video_loaded.emit(None)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minian UI 流程工具")
        
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(100, 100, int(screen.width() * 0.8), int(screen.height() * 0.8))

        self.data_manager = None
        self.pipeline = None
        self.current_worker = None
        self.current_mode = None
        self.video_loader_thread = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.setup_start_page()

    def setup_start_page(self):
        self.setWindowTitle("Minian UI 流程工具 - 选择流程")
        self.clear_layout(self.main_layout)

        start_page_layout = QVBoxLayout()
        start_page_layout.addStretch()

        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Minian流程", "新流程"])
        mode_layout.addWidget(QLabel("选择流程:"))
        mode_layout.addWidget(self.mode_combo)
        
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请输入或选择项目文件夹路径")
        select_btn = QPushButton("选择文件夹")
        select_btn.clicked.connect(self.select_project_folder)
        path_layout.addWidget(QLabel("项目路径:"))
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(select_btn)
        
        # 添加正则表达式输入框
        regex_layout = QHBoxLayout()
        self.regex_edit = QLineEdit()
        self.regex_edit.setPlaceholderText("自动生成的视频文件正则表达式")
        regex_layout.addWidget(QLabel("视频文件名模式:"))
        regex_layout.addWidget(self.regex_edit)
        
        load_btn = QPushButton("加载项目")
        load_btn.clicked.connect(self.load_selected_flow)
        
        container = QFrame()
        container_layout = QVBoxLayout(container)
        container_layout.addLayout(mode_layout)
        container_layout.addLayout(path_layout)
        container_layout.addLayout(regex_layout)
        container_layout.addWidget(load_btn)
        
        start_page_layout.addWidget(container, alignment=Qt.AlignCenter)
        start_page_layout.addStretch()
        
        self.main_layout.addLayout(start_page_layout)

    def select_project_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择项目文件夹")
        if folder_path:
            self.path_edit.setText(folder_path)
            self.generate_regex_pattern(folder_path)

    def generate_regex_pattern(self, folder_path: str):
        """
        根据文件夹内的所有.avi文件自动生成推荐的正则表达式。
        """
        avi_files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith('.avi')])
        
        if not avi_files:
            self.regex_edit.setText("")
            return

        # 获取所有文件名
        filenames = [os.path.basename(f) for f in avi_files]
        
        # 计算最长公共前缀和后缀
        prefix = _longest_common_prefix(filenames)
        suffix = _longest_common_suffix(filenames)
        
        # 移除前缀和后缀，剩下中间部分
        middle_parts = [f[len(prefix):len(f) - len(suffix)] for f in filenames]
        
        # 如果中间部分都是数字，则认为是可变部分
        is_all_digits = all(p.isdigit() for p in middle_parts)
        
        if is_all_digits:
            # 拼接正则表达式
            # 使用 re.escape 来转义前缀和后缀中的特殊字符
            pattern = f"{re.escape(prefix)}[0-9]+{re.escape(suffix)}$"
        else:
            # 如果中间部分不全为数字，则退回到只使用第一个文件的旧逻辑
            first_file = filenames[0]
            base_name = re.sub(r'\d+', '[0-9]+', first_file)
            pattern = re.escape(base_name)
            pattern = pattern.replace('\\[0-9\\]\\+', '[0-9]+')
            
        self.regex_edit.setText(pattern)


    def load_selected_flow(self):
        project_path = self.path_edit.text()
        if not project_path or not os.path.exists(project_path):
            QMessageBox.warning(self, "警告", "请选择一个有效的项目文件夹。")
            return
        
        regex_pattern = self.regex_edit.text()
        if not regex_pattern:
            QMessageBox.warning(self, "警告", "请输入有效的视频文件名模式。")
            return
        
        selected_flow = self.mode_combo.currentText()
        self.current_mode = selected_flow
        
        self.data_manager = DataManager(project_path)
        self.data_manager.initialize_repo()

        if selected_flow == "Minian流程":
            QMessageBox.information(self, "项目已加载", "项目数据已加载。正在后台预加载视频数据，请稍候...")
            self.video_loader_thread = VideoLoader(project_path, regex_pattern)
            self.video_loader_thread.video_loaded.connect(self.on_video_loaded)
            self.video_loader_thread.start()
        elif selected_flow == "新流程":
            self.setup_new_flow_ui()

    def on_video_loaded(self, varr):
        if varr is not None:
            self.pipeline = MinianPipeline(self.data_manager, varr)
            self.setup_minian_flow_ui()
        else:
            QMessageBox.critical(self, "错误", "视频加载失败，请检查路径和文件名模式。")
            self.setup_start_page()

    def setup_minian_flow_ui(self):
        self.setWindowTitle("Minian UI 流程工具 - Minian流程")
        self.clear_layout(self.main_layout)
        
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)
        
        self.steps = natsorted(STEP_NAMES.keys())

        self.navigation = StepNavigation(self.steps)
        self.main_layout.addWidget(self.navigation)
        self.navigation.nav_combo.currentIndexChanged.connect(self.stacked_widget.setCurrentIndex)
        self.navigation.next_button.clicked.connect(self.next_step)
        self.navigation.prev_button.clicked.connect(self.prev_step)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("QProgressBar {text-align: center;}")
        self.main_layout.addWidget(self.progress_bar)

        self.setup_steps_ui()
        
        # 切换到步骤0，并自动加载视频
        self.stacked_widget.setCurrentIndex(0)
        param_panel = self.find_param_panel(self.stacked_widget.currentWidget())
        if param_panel and hasattr(param_panel, "viz_widget"):
            viz_widget = param_panel.viz_widget
            if isinstance(viz_widget, VideoPlayer):
                viz_widget.load_video_data(self.pipeline.get_step_result("raw_video"))

    def setup_new_flow_ui(self):
        self.setWindowTitle("Minian UI 流程工具 - 新流程")
        self.clear_layout(self.main_layout)
        
        new_flow_page = QWidget()
        new_flow_layout = QVBoxLayout(new_flow_page)
        new_flow_layout.addStretch()
        new_flow_layout.addWidget(QLabel("<h1>新流程界面占位符</h1><p>请在此处添加您的新流程UI</p>", alignment=Qt.AlignCenter))
        new_flow_layout.addWidget(QPushButton("返回主菜单", clicked=self.setup_start_page))
        new_flow_layout.addStretch()
        
        self.main_layout.addWidget(new_flow_page)

    def setup_steps_ui(self):
        while self.stacked_widget.count():
            widget = self.stacked_widget.currentWidget()
            self.stacked_widget.removeWidget(widget)
            widget.deleteLater()
            
        for step_id, step_name in STEP_NAMES.items():
            page = self.create_step_page(step_id, step_name)
            self.stacked_widget.addWidget(page)
    
    def create_step_page(self, step_id: str, step_name: str) -> QWidget:
        page = QWidget()
        page_layout = QHBoxLayout(page)
        
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel(f"<h2>{step_name}</h2>"))
        left_layout.addSpacing(10)

        param_panel = ParamPanel(step_name, self.pipeline.step_params_map.get(step_id, {}))
        left_layout.addWidget(param_panel)
        
        run_btn = QPushButton("运行当前步骤")
        run_btn.clicked.connect(lambda: self.run_current_step(param_panel.get_params()))
        left_layout.addWidget(run_btn)
        
        left_layout.addStretch()
        
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        
        viz_widget = None
        if step_id in ["step0", "step1_1", "step1_2", "step1_3"]:
            viz_widget = VideoPlayer()
        elif step_id == "step2":
            viz_widget = ComparisonVideoPlayer()
        elif step_id in ["step3_1", "step3_2", "step3_3", "step3_4", "step4_1", "step4_2"]:
            viz_widget = CNMFVisualizer()
        
        if viz_widget:
            right_layout.addWidget(viz_widget)
            param_panel.viz_widget = viz_widget
        
        right_layout.addStretch()
        
        page_layout.addWidget(left_panel)
        page_layout.addWidget(right_panel)
        
        return page

    def run_current_step(self, params: dict):
        if self.current_mode != "Minian流程":
            QMessageBox.information(self, "信息", "当前流程不支持此操作。")
            return
            
        if not self.pipeline:
            QMessageBox.warning(self, "警告", "请先加载项目文件夹。")
            return
        
        current_idx = self.stacked_widget.currentIndex()
        step_id = self.steps[current_idx]
        
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.quit()
            self.current_worker.wait()
        
        self.progress_bar.setValue(0)
        self.current_worker = ProgressWorker(self.pipeline, step_id, params)
        self.current_worker.progress_updated.connect(self.progress_bar.setValue)
        self.current_worker.finished.connect(lambda: self.on_step_finished(step_id))
        self.current_worker.start()

    def on_step_finished(self, step_id: str):
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "完成", f"步骤 {STEP_NAMES[step_id]} 已完成。")
        
        result = self.pipeline.get_step_result(step_id)
        self.update_ui_on_step_finish(step_id, result)

    def update_ui_on_step_finish(self, step_id: str, result):
        current_page = self.stacked_widget.currentWidget()
        param_panel = self.find_param_panel(current_page)
        if not param_panel or not hasattr(param_panel, "viz_widget"):
            return

        viz_widget = param_panel.viz_widget
        
        if step_id in ["step0", "step1_1", "step1_2", "step1_3"]:
            if isinstance(viz_widget, VideoPlayer):
                viz_widget.load_video_data(result)
        elif step_id == "step2":
            if isinstance(viz_widget, ComparisonVideoPlayer):
                before_video = self.pipeline.get_step_result("varr_processed")
                after_video = self.pipeline.get_step_result("varr_mc")
                viz_widget.load_video_data(before_video, after_video)
        elif step_id in ["step3_1", "step3_2", "step3_3", "step3_4", "step4_1", "step4_2"]:
            if isinstance(viz_widget, CNMFVisualizer):
                A = self.pipeline.get_step_result("A")
                C = self.pipeline.get_step_result("C")
                viz_widget.update_all(A, C)
    
    def find_param_panel(self, parent_widget: QWidget) -> Optional[ParamPanel]:
        return parent_widget.findChild(ParamPanel)
    
    def next_step(self):
        idx = self.stacked_widget.currentIndex()
        if idx < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(idx + 1)
        self.navigation.nav_combo.setCurrentIndex(self.stacked_widget.currentIndex())

    def prev_step(self):
        idx = self.stacked_widget.currentIndex()
        if idx > 0:
            self.stacked_widget.setCurrentIndex(idx - 1)
        self.navigation.nav_combo.setCurrentIndex(self.stacked_widget.currentIndex())

    def clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())


class ProgressWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, pipeline: MinianPipeline, step_id: str, params: dict):
        super().__init__()
        self.pipeline = pipeline
        self.step_id = step_id
        self.params = params
    
    def run(self):
        try:
            self.pipeline.run_step(self.step_id, self.params, progress_callback=self.progress_updated.emit)
        finally:
            self.finished.emit()