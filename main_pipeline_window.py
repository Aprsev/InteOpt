import os
import json
import sys
import traceback
from typing import Dict, Any, Optional, List, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, 
    QComboBox, QMessageBox, QGroupBox, QSpinBox, QDoubleSpinBox, QSlider, 
    QTextEdit, QFormLayout, QGridLayout, QScrollArea
)
from PyQt5 import QtWidgets
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QMutex, QLocale, QObject
)
from PyQt5.QtGui import QPixmap, QImage, QColor
import numpy as np
import xarray as xr
import pandas as pd
from threading import Lock

# 导入核心组件
from minian_processor import MinianProcessor # 假设已实现
from minian_core.visualization import (
    get_normalized_video_frame,  
    create_seeds_visualization, create_pnr_refine_plot, 
    create_exploration_plot, create_cnmf_update_plot, normalize_frame,
    create_mc_max_projection_comparison,
    create_init_visualization_plot 
)

# =========================================================================
# 1. 步骤定义和状态管理
# =========================================================================

# 定义流程步骤
# (步骤ID, 中文名称, 代码名称, 可视化类型)
PIPELINE_STEPS: List[Tuple[int, str, str, str]] = [
    (1, "加载视频与去除光晕", "load_video_1", "video"),
    (3, "降噪 (时域/空域)", "denoise", "video"),
    (4, "去除背景", "background_removal", "video"), # 调整顺序，与实际处理更一
    (5, "运动校正", "motion_correction", "split_video"),
    (6, "生成过完备种子点", "seeds_init", "seeds"),
    (7, "噪声频率探索", "noise_freq_exploration", "curve_exploration"),
    (8, "信噪比精修", "peak_noise_ratio_refine", "seeds"),
    (9, "KS检验精修", "ks_refine", "seeds"),
    (10, "合并种子点", "merge_seeds", "seeds"),
    (11, "初始化可视化", "visualization_init", "cnmf_init"),
    (12, "初次空间更新 (参数探索)", "first_spatial_update_explore", "exploration"),
    (13, "初次空间更新 (执行)", "first_spatial_update_exec", "cnmf_update"),
    (14, "初次时间更新 (参数探索)", "first_temporal_update_explore", "exploration"),
    (15, "初次时间更新 (执行)", "first_temporal_update_exec", "cnmf_update"),
    (16, "第二次空间更新", "second_spatial_update", "cnmf_update"),
    (17, "第二次时间更新", "second_temporal_update", "cnmf_update"),
    (18, "数据保存", "save_data", "none"),
]

# 状态颜色
STEP_STATUS_COLORS = {
    "未运行": "lightgray",
    "运行中": "yellow",
    "已完成": "lightgreen",
    "有缓存": "lightblue",
    "错误": "red",
}

# =========================================================================
# 2. 线程工作器 (Worker Thread)
# =========================================================================

class WorkerSignals(QObject):
    """定义 WorkerThread 发送给主线程的信号。"""
    finished = pyqtSignal()
    error = pyqtSignal(str, str, str) # (步骤名称, 错误类型, 错误信息)
    status_update = pyqtSignal(str, str) # (步骤代码名, 状态)
    step_result = pyqtSignal(str, object) # (步骤代码名, 结果数据)
    log_message = pyqtSignal(str) # 终端/日志输出

class WorkerThread(QThread):
    """
    负责在后台运行 Minian 耗时计算的线程。
    """
    def __init__(self, processor: MinianProcessor, signals: WorkerSignals):
        super().__init__()
        self.processor = processor
        self.signals = signals
        # self.mutex = QMutex()
        self.mutex = Lock()  
        self._is_running = False
        self._current_task: Optional[Tuple[str, bool]] = None # (步骤代码名, 是否为运行全部)
        self.all_steps_list = [name for _, _, name, _ in PIPELINE_STEPS]

    def set_task(self, step_name: str, run_all: bool = False, run_to: Optional[str] = None):
        """设置要运行的步骤或整个流程。
        
        参数:
            step_name: 开始运行的步骤名称
            run_all: 是否运行所有步骤
            run_to: 运行到的目标步骤名称
        """
        with self.mutex:
            self._current_task = (step_name, run_all, run_to)
            self._is_running = True

    def run(self):
        """线程主循环，执行 Minian 步骤。"""
        while True:
            step_name, run_all, run_to = None, False, None
            with self.mutex:
                if not self._is_running or self._current_task is None:
                    break
                step_name, run_all, run_to = self._current_task
                self._current_task = None # 清除当前任务，准备接收下一个任务

            if run_all:
                self._run_all_steps_from(step_name, run_to)
            else:
                self._run_single_step(step_name)
        
        self.signals.finished.emit()
        self.signals.log_message.emit("--- 后台线程执行完毕 ---")


    def _run_single_step(self, step_name: str):
        """执行单个 Minian 步骤。"""
        self.signals.log_message.emit(f"\n--- 开始运行步骤: {step_name} ---")
        self.signals.status_update.emit(step_name, "运行中")

        try:
            # 动态调用 processor 上的 run_step_X 方法
            run_func = getattr(self.processor, f"run_{step_name}")
            result = run_func()
            
            # 发送结果和状态更新
            self.signals.step_result.emit(step_name, result)
            self.signals.status_update.emit(step_name, "已完成")
            self.signals.log_message.emit(f"--- 步骤 {step_name} 成功完成 ---")

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            self.signals.status_update.emit(step_name, "错误")
            self.signals.error.emit(step_name, error_type, error_msg)
            self.signals.log_message.emit(f"--- 步骤 {step_name} 运行失败 ({error_type}) ---")

    def _run_all_steps_from(self, start_step_name: str, end_step_name: Optional[str] = None):
        """从指定步骤开始运行到结束步骤。
        
        参数:
            start_step_name: 开始运行的步骤名称
            end_step_name: 结束运行的步骤名称 (None表示运行到最后)
        """
        try:
            start_index = self.all_steps_list.index(start_step_name)
            end_index = len(self.all_steps_list) if end_step_name is None else self.all_steps_list.index(end_step_name)
            
            # 确保end_index不小于start_index
            if end_index < start_index:
                end_index = start_index
                self.signals.log_message.emit(f"警告: 目标步骤 {end_step_name} 在开始步骤 {start_step_name} 之前，将只运行开始步骤")
            
            # 连续运行所有步骤直到结束步骤
            for i in range(start_index, end_index + 1):
                step_name = self.all_steps_list[i]
                
                # 检查是否应该停止
                if not self._is_running:
                    self.signals.log_message.emit(f"--- 运行到步骤 {step_name} 被中断 ---")
                    break
                    
                self._run_single_step(step_name)
                
                # 检查步骤是否成功完成
                if self.processor.get_step_status(step_name) != "已完成":
                    self.signals.log_message.emit(f"步骤 {step_name} 未成功完成，停止运行")
                    break
                    
                # 检查是否到达目标步骤
                if i == end_index:
                    self.signals.log_message.emit(f"--- 成功运行到目标步骤 {step_name} ---")
                    
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            self.signals.error.emit("_run_all_steps_from", error_type, error_msg)
            self.signals.log_message.emit(f"运行到指定步骤失败: {error_type} - {error_msg}")
        finally:
            with self.mutex:
                self._is_running = False
                self._current_task = None


# =========================================================================
# 3. PyQt5 主窗口
# =========================================================================

class MainPipelineWindow(QWidget):
    """
    Minian UI 主流程窗口，包含参数调整、步骤控制和可视化显示。
    """
    
    def __init__(self, processor: MinianProcessor, pipeline_mode: str, regex_pattern: str):
        super().__init__()
        QLocale.setDefault(QLocale(QLocale.Chinese, QLocale.China))
        self.setWindowTitle(f"Minian UI - 主流程 ({pipeline_mode})")
        self.setGeometry(100, 100, 1200, 800)
        
        self.processor = processor
        self.regex_pattern = regex_pattern
        self._dynamic_widgets: Dict[str, Dict[str, Any]] = {} 
        
        # 流程状态
        self.steps_map = {name: (id, cn_name, vis_type) for id, cn_name, name, vis_type in PIPELINE_STEPS}
        self.step_names = [name for _, _, name, _ in PIPELINE_STEPS]
        self.current_step_name = self.step_names[0]
        self.steps_status: Dict[str, str] = {name: "未运行" for name in self.step_names}
        self.steps_results: Dict[str, Any] = {} # 存储结果用于可视化
        
        # 可视化状态
        self.current_frame = 0
        self.total_frames = 1 # 启动时默认为 1
        self.visualization_timer = None # 用于视频播放的 QTimer
        
        self.init_ui()
        self.init_worker_thread()
        self.update_step_list_widget()
        self.update_parameters_panel()
        self.log_output.append(f"初始化成功。视频文件夹: {processor.dpath}")
        self.log_output.append(f"当前流程: {pipeline_mode}")
        
    def init_ui(self):
        """初始化 UI 布局。"""
        main_layout = QHBoxLayout(self)
        
        # --- 左侧控制面板 ---
        left_panel = QWidget()
        left_panel.setFixedWidth(400)  # 设置左侧控制面板的宽度
        left_layout = QVBoxLayout(left_panel)
        
        # 1. 流程控制/步骤选择
        control_group = QGroupBox("流程控制与步骤选择")
        control_layout = QVBoxLayout(control_group)
        
        # 配置文件选择 (左上角要求)
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("当前配置库:"))
        config_layout.addWidget(QLabel(os.path.basename(self.processor.config_path)))
        control_layout.addLayout(config_layout)
        
        # 步骤选择下拉框
        self.step_select_combo = QComboBox()
        self.step_select_combo.currentIndexChanged.connect(self.switch_step_visualization)
        control_layout.addWidget(QLabel("选择步骤查看效果/重运行:"))
        control_layout.addWidget(self.step_select_combo)
        
        # 步骤状态列表（用 GroupBox 模拟列表，方便着色）
        self.step_list_group = QGroupBox("Minian 步骤状态")
        self.step_list_scroll = QScrollArea()  # 创建滚动区域
        self.step_list_scroll.setWidgetResizable(True)  # 允许滚动区域调整大小
        self.step_list_content = QWidget()  # 创建滚动区域的内容
        self.step_list_layout = QVBoxLayout(self.step_list_content)  # 创建内容的布局
        self.step_list_scroll.setWidget(self.step_list_content)  # 将内容设置到滚动区域
        self.step_list_group_layout = QVBoxLayout(self.step_list_group)  # 创建步骤状态组的布局
        self.step_list_group_layout.addWidget(self.step_list_scroll)  # 将滚动区域添加到组布局
        control_layout.addWidget(self.step_list_group)
        
        # 2. 参数设置面板
        self.param_group = QGroupBox("当前步骤参数设置")
        self.param_content = QWidget()
        self.param_form_layout = QFormLayout(self.param_content)
        
        param_layout = QVBoxLayout(self.param_group)
        param_layout.addWidget(self.param_content)
        
        # 操作按键
        btn_layout = QGridLayout()
        self.btn_prev = QPushButton("上一步 (<)")
        self.btn_prev.clicked.connect(lambda: self.switch_step(-1))
        self.btn_run_current = QPushButton("运行当前步骤 (▶)")
        self.btn_run_current.clicked.connect(self.run_current_step)
        self.btn_next = QPushButton("下一步 (>)")
        self.btn_next.clicked.connect(lambda: self.switch_step(1))
        self.btn_run_all = QPushButton("运行所有步骤 (▶▶)")
        self.btn_run_all.setStyleSheet("background-color: #fdd835;")  # 醒目颜色
        self.btn_run_all.clicked.connect(self.run_all_steps)
        
        btn_layout.addWidget(self.btn_prev, 0, 0)
        btn_layout.addWidget(self.btn_run_current, 0, 1)
        btn_layout.addWidget(self.btn_next, 0, 2)
        btn_layout.addWidget(self.btn_run_all, 1, 0, 1, 3)
        
        # 添加运行到指定步骤功能
        self.step_combo = QComboBox()
        self.step_combo.addItems([f"{id}: {cn_name}" for id, cn_name, _, _ in PIPELINE_STEPS])
        self.run_to_btn = QPushButton("运行到指定步骤")
        self.run_to_btn.clicked.connect(self.run_to_selected_step)
        btn_layout.addWidget(self.step_combo, 2, 0, 1, 2)
        btn_layout.addWidget(self.run_to_btn, 2, 2)
        
        param_btn_layout = QVBoxLayout()
        param_btn_layout.addWidget(self.param_group)
        param_btn_layout.addLayout(btn_layout)
        
        left_layout.addWidget(control_group)
        left_layout.addLayout(param_btn_layout)
        left_layout.addStretch(1)
        
        main_layout.addWidget(left_panel)
        
        # --- 右侧可视化窗格 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 1. 可视化显示区域
        vis_group = QGroupBox("实时可视化")
        vis_layout = QVBoxLayout(vis_group)
        
        # 视频/图像显示 QLabel
        self.vis_label = QLabel("请运行第一步以加载视频...")
        self.vis_label.setAlignment(Qt.AlignCenter)
        self.vis_label.setMinimumSize(700, 450)
        self.vis_label.setStyleSheet("border: 1px solid black;")
        vis_layout.addWidget(self.vis_label)
        
        # 视频控制条
        video_control_group = QGroupBox("视频/数据播放控制")
        video_control_layout = QGridLayout(video_control_group)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.valueChanged.connect(self.update_frame_from_slider)
        self.slider.setEnabled(False)
        
        self.frame_label = QLabel(f"帧: 0 / {self.total_frames}")
        self.play_pause_btn = QPushButton("开始 (▶)")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.rewind_btn = QPushButton("快退 (<<)")
        self.rewind_btn.clicked.connect(lambda: self.seek_frame(-30))
        self.forward_btn = QPushButton("快进 (>>)")
        self.forward_btn.clicked.connect(lambda: self.seek_frame(30))
        
        video_control_layout.addWidget(self.slider, 0, 0, 1, 4)
        video_control_layout.addWidget(self.frame_label, 1, 0)
        video_control_layout.addWidget(self.rewind_btn, 1, 1)
        video_control_layout.addWidget(self.play_pause_btn, 1, 2)
        video_control_layout.addWidget(self.forward_btn, 1, 3)
        
        vis_layout.addWidget(video_control_group)
        right_layout.addWidget(vis_group)
        
        # 2. 终端输出/日志
        log_group = QGroupBox("终端输出与日志")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        right_layout.addWidget(log_group)
        
        main_layout.addWidget(right_panel)

    # =========================================================================
    # 4. 线程和信号初始化
    # =========================================================================
    
    def init_worker_thread(self):
        """初始化后台工作线程。"""
        self.worker_signals = WorkerSignals()
        self.worker_thread = WorkerThread(self.processor, self.worker_signals)
        
        # 连接信号到主线程槽函数
        self.worker_signals.log_message.connect(self.update_log)
        self.worker_signals.status_update.connect(self._update_step_status)
        self.worker_signals.error.connect(self.handle_worker_error)
        self.worker_signals.step_result.connect(self.handle_step_result)
        self.worker_signals.finished.connect(self.on_worker_finished)
        
        self.is_running_all = False
        
    def _set_ui_running_state(self, running: bool):
        """设置 UI 控件的启用/禁用状态。"""
        self.btn_run_current.setEnabled(not running)
        self.btn_run_all.setEnabled(not running)
        self.btn_prev.setEnabled(not running)
        self.btn_next.setEnabled(not running)
        self.step_select_combo.setEnabled(not running)
        
    def on_worker_finished(self):
        """工作线程结束后调用。"""
        self._set_ui_running_state(False)
        
        # 检查是否有自动运行任务
        if hasattr(self, '_auto_run_target'):
            target_step = self._auto_run_target
            current_step = self.current_step_name
            
            # 检查是否到达目标步骤
            if current_step == target_step:
                QMessageBox.information(self, "完成", f"已成功运行到目标步骤 {self.steps_map[target_step][1]}")
                delattr(self, '_auto_run_target')
            else:
                # 继续自动运行
                self._auto_run_to_step(target_step)
        elif self.is_running_all:
            QMessageBox.information(self, "流程完成", "所有步骤已运行完毕！")
            self.is_running_all = False

    # =========================================================================
    # 5. 流程控制和参数管理
    # =========================================================================

    def switch_step(self, direction: int):
        """
        根据方向切换到上一步或下一步。
        
        修正逻辑：前进时，检查当前步骤是否已完成，而不是检查下一步是否未运行。
        """
        # 注意: 假设您已经在文件顶部导入了 QMessageBox，如果未导入，请添加：
        # from PyQt5.QtWidgets import QMessageBox 
        
        # 获取当前步骤的索引和名称
        current_idx = self.step_names.index(self.current_step_name)
        current_step_name = self.current_step_name
        new_idx = current_idx + direction
        
        # 检查新索引是否在有效范围内
        if 0 <= new_idx < len(self.step_names):
            if direction > 0:
                current_status = self.steps_status.get(current_step_name)
                
                # 假设步骤成功后的状态是 "已完成" 或 "成功"
                # 任何其它状态 (如 "未运行", "运行中", "错误") 都应阻止前进
                if current_status not in ["已完成", "成功"]:
                    
                    # 给出更友好的提示信息
                    if current_status == "运行中":
                        msg = f"当前步骤【{self.steps_map[current_step_name][1]}】仍在运行中，请等待其完成。"
                    else:
                        msg = f"当前步骤【{self.steps_map[current_step_name][1]}】尚未成功运行。请点击 '运行当前步骤' 或等待完成。"
                        
                    QMessageBox.warning(self, "警告", msg)
                    return

            new_step_name = self.step_names[new_idx]
            self.current_step_name = new_step_name
            self.update_parameters_panel()
            self.update_step_list_widget(force_select=True)
            self.visualize_current_step()
            
    def switch_step_visualization(self, index: int):
        """
        通过下拉框切换步骤。
        """
        step_cn_name = self.step_select_combo.itemText(index)
        # 从中文名获取代码名
        for name in self.step_names:
            if self.steps_map[name][1] == step_cn_name:
                self.current_step_name = name
                break
                
        self.update_parameters_panel()
        self.update_step_list_widget(force_select=True)
        self.visualize_current_step()

    def run_current_step(self):
        """
        运行当前选定步骤的逻辑。
        1. 检查参数更新并保存到配置文件。
        2. 标记后续步骤为 '未运行' 并清除缓存。
        3. 启动工作线程执行当前步骤。
        """
        step_name = self.current_step_name
        
        # 1. 检查并保存参数
        if self._check_and_save_parameters(step_name):
            self.log_output.append(f"参数已更新并保存到配置库。")
            
        # 2. 标记后续步骤为 '未运行' 并清除缓存
        current_idx = self.step_names.index(step_name)
        for i in range(current_idx + 1, len(self.step_names)):
            subsequent_step = self.step_names[i]
            if self.steps_status[subsequent_step] not in ["未运行", "错误"]:
                self.steps_status[subsequent_step] = "未运行"
                # TODO: 实际的缓存清除操作（例如：删除 Zarr 文件）
                self.log_output.append(f"标记步骤 {self.steps_map[subsequent_step][1]} 为 '未运行' 并清除缓存。")
        self.update_step_list_widget()

        # 3. 启动线程
        self._set_ui_running_state(True)
        self.worker_thread.set_task(step_name, run_all=False)
        if not self.worker_thread.isRunning():
            self.worker_thread.start()

    def run_all_steps(self):
        """
        从当前步骤开始，使用库中参数运行所有后续步骤。
        """
        step_name = self.current_step_name
        
        reply = QMessageBox.question(self, '确认运行全部', 
            f"将从步骤 '{self.steps_map[step_name][1]}' 开始运行所有后续步骤。\n将使用库中已保存的参数进行计算。\n确认开始吗？", 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
        if reply == QMessageBox.No:
            return
            
        self.is_running_all = True
        self._set_ui_running_state(True)
        self.worker_thread.set_task(step_name, run_all=True)
        if not self.worker_thread.isRunning():
            self.worker_thread.start()

    def run_to_selected_step(self):
        """
        从当前步骤自动运行到选择的步骤，模拟按键操作。
        """
        current_step = self.current_step_name
        selected_text = self.step_combo.currentText()
        step_id = int(selected_text.split(":")[0])
        target_step = next(name for id, _, name, _ in PIPELINE_STEPS if id == step_id)
        
        # 检查是否已经到达目标步骤
        if current_step == target_step:
            QMessageBox.information(self, "提示", f"当前已经是步骤 {self.steps_map[target_step][1]}，无需运行")
            return
            
        # 检查目标步骤是否在当前步骤之前
        current_idx = self.step_names.index(current_step)
        target_idx = self.step_names.index(target_step)
        if target_idx < current_idx:
            QMessageBox.warning(self, "警告", "目标步骤必须在当前步骤之后！")
            return
            
        # 确认对话框
        reply = QMessageBox.question(
            self, '确认运行到指定步骤',
            f"将从步骤 '{self.steps_map[current_step][1]}' 自动运行到步骤 '{self.steps_map[target_step][1]}'。\n确认开始吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
        if reply == QMessageBox.No:
            return
            
        # 设置自动运行目标
        self._auto_run_target = target_step
        # 禁用所有控制按钮
        self._set_ui_running_state(True)
        # 开始自动运行流程
        self._auto_run_to_step(target_step)
        
    def _auto_run_to_step(self, target_step: str):
        """自动运行到指定步骤的内部实现"""
        current_step = self.current_step_name
        
        # 检查是否已经到达目标步骤
        if current_step == target_step:
            QMessageBox.information(self, "完成", f"已成功运行到目标步骤 {self.steps_map[target_step][1]}")
            self._set_ui_running_state(False)
            if hasattr(self, '_auto_run_target'):
                delattr(self, '_auto_run_target')
            return
            
        # 检查当前步骤状态
        current_status = self.steps_status.get(current_step)
        
        if current_status != "已完成":
            # 如果当前步骤未完成，先运行当前步骤
            self.log_output.append(f"自动运行: 正在运行当前步骤 {current_step}...")
            self.worker_thread.set_task(current_step)
            if not self.worker_thread.isRunning():
                self.worker_thread.start()
        else:
            # 如果当前步骤已完成，切换到下一步
            current_idx = self.step_names.index(current_step)
            next_step = self.step_names[current_idx + 1]
            self.log_output.append(f"自动运行: 切换到下一步 {next_step}...")
            self.current_step_name = next_step
            self.update_step_list_widget(force_select=True)
            self.update_parameters_panel()
            
            # 递归调用继续运行
            self._auto_run_to_step(target_step)

    def _check_and_save_parameters(self, step_name: str) -> bool:
        """
        [修复版] 读取 UI 中的参数，与库中参数对比，如有更新则保存。
        """
        new_params = {}
        current_config = self.processor.get_step_params(step_name)
        
        print(f"--- 开始读取步骤 {step_name} 的 UI 参数 ---")

        # 遍历 QFormLayout 中的所有行
        for i in range(self.param_form_layout.rowCount()):
            
            # 1. 获取 Label 和 Field
            label_item = self.param_form_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.param_form_layout.itemAt(i, QFormLayout.FieldRole)
            
            if label_item is None or field_item is None:
                continue
                
            label_widget = label_item.widget()
            field_widget = field_item.widget()
            
            if label_widget is None or field_widget is None:
                continue

            # 2. 获取参数名
            # 注意：这里要处理可能存在的动态包装器 (Widget Wrapper)
            # 如果是动态参数，field_widget 可能是一个包含 Label 和 Editor 的 QWidget (HBoxLayout)
            param_key = label_widget.text().split('(')[0].strip()
            
            # 如果是动态行（我们在 update_parameters_panel 里创建的 row_widget）
            # 这种情况在这个循环里可能直接处理不到内部的 Editor，
            # 因为 update_parameters_panel 里的动态行是 addRow(row_widget)，没有 LabelRole
            # 所以这里的逻辑主要处理静态参数和 method 选择器。
            
            # --- 处理常规控件 ---
            val = None
            if isinstance(field_widget, QLineEdit):
                val = field_widget.text()
            elif isinstance(field_widget, (QSpinBox, QDoubleSpinBox)):
                val = field_widget.value()
            elif isinstance(field_widget, QComboBox):
                val = field_widget.currentText()
            
            if val is not None:
                new_params[param_key] = val

        # 3. [关键修复] 专门处理动态控件列表 self._dynamic_widgets
        # 因为 update_parameters_panel 中，动态参数被放进了 self._dynamic_widgets
        for key, item in self._dynamic_widgets.items():
            wrapper = item['widget']
            # 查找 wrapper 内部的编辑器控件
            # wrapper layout: [0: Label, 1: Editor]
            layout = wrapper.layout()
            if layout and layout.count() > 1:
                editor = layout.itemAt(1).widget()
                val = None
                
                if isinstance(editor, QLineEdit):
                    val = editor.text()
                elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
                    val = editor.value()
                elif isinstance(editor, QComboBox):
                    val = editor.currentText()
                
                if val is not None:
                    new_params[key] = val
                    print(f"读取动态参数: {key} = {val}")

        # 4. 类型转换与保存
        final_params_to_save = {}
        has_changed = False
        
        for key, str_val in new_params.items():
            # 尝试恢复原始类型
            old_val = current_config.get(key)
            
            # 转换逻辑
            try:
                if old_val is not None:
                    target_type = type(old_val)
                    if target_type == bool:
                        # 处理 "True"/"False" 字符串
                        final_val = str(str_val).lower() == 'true'
                    elif target_type == int:
                        final_val = int(float(str_val)) # 处理 "1.0" 转 int
                    elif target_type == float:
                        final_val = float(str_val)
                    elif target_type in (list, dict):
                        if isinstance(str_val, str):
                            final_val = json.loads(str_val)
                        else:
                            final_val = str_val
                    else:
                        final_val = str_val
                else:
                    # 如果配置文件里没有这个值（新增的），尝试智能推断
                    if isinstance(str_val, str):
                        if str_val.lower() == 'true': final_val = True
                        elif str_val.lower() == 'false': final_val = False
                        elif str_val.replace('.','',1).isdigit(): 
                            final_val = float(str_val) if '.' in str_val else int(str_val)
                        else:
                            final_val = str_val
                    else:
                        final_val = str_val

                final_params_to_save[key] = final_val
                
                # 检查变更
                if str(final_val) != str(old_val):
                    print(f"参数变更: {key} | 旧: {old_val} -> 新: {final_val}")
                    has_changed = True
                    
            except Exception as e:
                print(f"参数转换错误 {key}: {e}")
                final_params_to_save[key] = str_val # 转换失败保留原值

        # 5. 强制保存
        # 即使 has_changed 为 False，为了保险起见（防止之前保存失败），建议也调用 update
        if final_params_to_save:
            self.processor.update_params(step_name, final_params_to_save)
            self.log_output.append(f"已更新步骤 {step_name} 的参数。")
            return True
            
        return False
        
    # =========================================================================
    # 6. UI 更新和日志
    # =========================================================================

    def update_log(self, message: str):
        """接收 WorkerThread 的日志消息并显示。"""
        self.log_output.append(message)
        
    def _update_step_status(self, step_name: str, status: str):
        """更新步骤状态，并触发 UI 刷新。"""
        self.steps_status[step_name] = status
        self.update_step_list_widget()
        
        # 如果是当前步骤状态更新，触发可视化更新
        if step_name == self.current_step_name and status == "已完成":
            self.visualize_current_step()

    def update_step_list_widget(self, force_select: bool = False):
        """刷新步骤列表和下拉框，显示状态和当前选择。"""
        # 清除现有控件
        while self.step_list_layout.count():
            item = self.step_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                
        self.step_select_combo.blockSignals(True)
        
        self.step_select_combo.clear()

        # 重新创建步骤状态标签
        for step_name in self.step_names:
            id, cn_name, vis_type = self.steps_map[step_name]
            status = self.steps_status.get(step_name, "未运行")
            color = STEP_STATUS_COLORS.get(status, "white")
            
            label = QLabel(f"{id}. {cn_name} ({status})")
            label.setStyleSheet(
                f"background-color: {color}; padding: 5px; border: 1px solid #ccc; "
                f"font-weight: {'bold' if step_name == self.current_step_name else 'normal'};"
            )
            self.step_list_layout.addWidget(label)
            
            # 只允许选择 '已完成' 的步骤进行重运行/查看
            if status == "已完成" or step_name == self.current_step_name:
                self.step_select_combo.addItem(cn_name)
            
            if step_name == self.current_step_name:
                # 确保下拉框始终包含当前步骤，即使它未完成
                if status != "已完成" and cn_name not in [self.step_select_combo.itemText(i) for i in range(self.step_select_combo.count())]:
                    self.step_select_combo.addItem(cn_name)
                    
                # 重新选择当前步骤
                idx = self.step_select_combo.findText(cn_name)
                if idx != -1:
                    # 这一行是触发信号的根源
                    self.step_select_combo.setCurrentIndex(idx)

        self.step_select_combo.blockSignals(False) 

        self.step_list_group.setTitle(f"Minian 步骤状态 (当前: {self.steps_map[self.current_step_name][1]})")
    def _get_mode_prefix(self, key: str) -> str:
        """根据键名获取模式前缀，例如 'fft_low_cut' -> 'fft'"""
        parts = key.split('_', 1) # 只在第一个 '_' 处分割
        # 确保只匹配非 'method' 且有下划线的参数
        return parts[0] if len(parts) > 1 and parts[0] != 'method' else ''

    # ------------------------------------------------------------------
    # 新增: 辅助方法 - 刷新可见性的槽函数 (通用化)
    # ------------------------------------------------------------------
    def _update_dynamic_param_visibility(self, step_name: str, new_method: str):
        """
        通用槽函数：在模式切换时，更新配置并控制参数的显示/隐藏。
        """
        # 1. 💥 关键修复 💥: 使用 self.processor 来更新配置
        self.processor.update_config_param(step_name, 'method', new_method)

        # 2. 控制参数可见性
        for key, item in self._dynamic_widgets.items():
            widget = item['widget']
            param_mode = item['mode']
            
            # 如果参数的前缀与新选择的 method 匹配，则显示；否则隐藏
            is_visible = (param_mode == new_method)
            widget.setVisible(is_visible)

        # 强制刷新布局
        self.param_group.update()
        
    # ------------------------------------------------------------------
    # 替换/修改: 核心参数面板更新函数 (整合了静态和动态逻辑)
    # ------------------------------------------------------------------
    def update_parameters_panel(self):
        """
        核心函数：更新整个参数面板，自动识别多模式步骤并处理动态显示。
        """
        step_name = self.current_step_name
        cn_name = self.steps_map[step_name][1]
        self.param_group.setTitle(f"参数设置: {cn_name} ({step_name})")
        
        # 清除现有参数控件
        while self.param_form_layout.rowCount() > 0:
            self.param_form_layout.removeRow(0)
        
        # 重置动态控件映射 (只在动态模式下使用)
        self._dynamic_widgets = {}
        
        try:
            params = self.processor.get_step_params(step_name)
        except Exception as e:
            self.log_output.append(f"错误: 无法加载步骤 {step_name} 的参数: {e}")
            return
            
        is_multi_mode = 'method' in params
        current_method = params.get('method', '')
        
        for key, value in params.items():
            editor = None
            label = QLabel(f"{key}")
            
            # --- 1. 处理模式选择器 ---
            if key == 'method':
                editor = QComboBox()
                # 从所有参数键中提取前缀作为模式选项
                all_modes = sorted(list(set(self._get_mode_prefix(k) for k in params.keys() if self._get_mode_prefix(k))))
                editor.addItems(all_modes)
                editor.setCurrentText(str(value))
                
                # 核心：连接信号到动态控制函数
                editor.currentTextChanged.connect(
                    lambda text: self._update_dynamic_param_visibility(step_name, text)
                )
                
                self.param_form_layout.addRow(label, editor)
                continue
                
            # --- 2. 创建编辑器 (复用您原始的代码逻辑) ---
            if isinstance(value, int):
                editor = QSpinBox()
                MAX_SPINBOX_INT = 1000000000 # 1e9, 确保设置的值能被容纳
                editor.setRange(-MAX_SPINBOX_INT, MAX_SPINBOX_INT) 
                editor.setValue(value)
            elif isinstance(value, float):
                editor = QDoubleSpinBox()
                MAX_FLOAT_RANGE = 1e18 # 足够大，且小于 float 的最大值
                editor.setMinimum(-MAX_FLOAT_RANGE)
                editor.setMaximum(MAX_FLOAT_RANGE)
                editor.setDecimals(4)
                editor.setValue(value)
            elif isinstance(value, bool):
                editor = QComboBox()
                editor.addItems(["True", "False"])
                editor.setCurrentText(str(value))
            elif isinstance(value, (list, dict)):
                editor = QLineEdit(json.dumps(value))
                editor.setToolTip("输入 JSON 格式的列表或字典")
            elif value is None:
                editor = QLineEdit("null")
            else:
                editor = QLineEdit(str(value))
               
                
            # TODO: 确保您已连接编辑器的信号到配置更新 (在 run_current_step 中有检查并保存的逻辑，这里可以省略，但最好在编辑时就更新配置)
            
            # --- 3. 动态模式步骤的特殊处理 ---
            if is_multi_mode:
                mode_prefix = self._get_mode_prefix(key)
                
                # 为模式特定参数创建包装器
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0) # 消除边距
                
                # 将 Label 和 Editor 放入包装器
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                row_layout.addWidget(label)
                row_layout.addWidget(editor)
                
                # 初始可见性设置
                if mode_prefix != current_method:
                    row_widget.setVisible(False)
                    
                # 存储控件信息
                self._dynamic_widgets[key] = {
                    'widget': row_widget,
                    'mode': mode_prefix
                }
                
                # 将包装器添加到布局中
                self.param_form_layout.addRow(row_widget)
                
            else:
                # 静态模式步骤，保持原有布局方式
                self.param_form_layout.addRow(label, editor)
    # =========================================================================
    # 7. 可视化和视频播放
    # =========================================================================

    def handle_worker_error(self, step_name: str, error_type: str, error_msg: str):
        """处理工作线程报告的错误。"""
        self._set_ui_running_state(False)
        self.log_output.append(f"*** 步骤 {self.steps_map[step_name][1]} 运行时发生错误: {error_type} ***")
        self.log_output.append(f"错误详情: {error_msg}")
        
        # 文件读取错误，弹出弹窗让用户修改路径
        if "FileNotFoundError" in error_type or "路径错误" in error_msg:
             QMessageBox.critical(self, "运行错误", 
                 f"步骤 '{self.steps_map[step_name][1]}' 发生文件读取错误或路径无效。\n错误信息: {error_msg}\n请修改左侧参数并再次运行当前步骤。",
                 QMessageBox.Ok)
        else:
             QMessageBox.critical(self, "运行错误", 
                 f"步骤 '{self.steps_map[step_name][1]}' 运行失败。\n错误类型: {error_type}\n详情已输出到日志。",
                 QMessageBox.Ok)


    def handle_step_result(self, step_name: str, result: Any):
        """处理步骤运行结果，存储并触发可视化更新。"""
        # 存储结果
        self.steps_results[step_name] = result
        self.log_output.append(f"步骤 {self.steps_map[step_name][1]} 结果已接收。{result}")
        # print(f"步骤 {self.steps_map[step_name][1]} 结果已接收。{result}")
        # 针对视频更新总帧数
        if isinstance(result, xr.DataArray) and 'frame' in result.dims:
            self.total_frames = result.sizes['frame']
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setEnabled(True)
            self.frame_label.setText(f"帧: {self.current_frame} / {self.total_frames}")

        temp_step_name = self.current_step_name
        
        # (B) 强制将 UI 的当前步骤设置回刚刚完成的步骤
        self.current_step_name = step_name
        # (C) 调用可视化刷新
        self.visualize_current_step()
        # (D) 恢复 UI 的当前步骤（如果它已经被线程切换了）
        self.current_step_name = temp_step_name
        # (E) 刷新步骤列表，确保高亮显示正确
        self.update_step_list_widget(force_select=True)
            
    def visualize_current_step(self):
        """
        根据当前选定的步骤，获取相应的可视化数据，并设置总帧数和滑块。
        最后调用 _update_visualization_frame 刷新显示。
        """
        step_name = self.current_step_name
        # 🚨 警告修复点：获取步骤状态并检查是否已完成
        status = self.steps_status.get(step_name)
        
        if status != "已完成":
            # 如果步骤不是“已完成”，则清空显示，禁用滑块，并立即退出
            self.total_frames = 1
            self.current_frame = 0
            self.slider.setMaximum(0)
            self.slider.setValue(0)
            self.slider.setEnabled(False)
            self.vis_label.setText(f"步骤 '{step_name}' 状态: {status}。请先运行此步骤以查看结果。")
            self.frame_label.setText("帧: 1 / 1")
            return # <<< 提前退出，避免调用 self.processor.get_varr_for_vis 触发警告
        # 1. 获取用于确定帧数的视频数组 (varr)
        # 我们依赖 MinianProcessor.get_varr_for_vis 来获取当前步骤的可视化背景视频数组。
        # 如果 get_varr_for_vis 返回 None，则视为非帧依赖步骤。
        varr = self.processor.get_varr_for_vis(step_name)

        # self.log_output.append(f"步骤 {self.steps_map[step_name][1]} 结果已获取。{varr}")
        if isinstance(varr, tuple) and len(varr) > 1:
            # 提取运动校正后的视频 varr_mc (第二个元素)
            self.log_output.append(f"DEBUG: 步骤 '{step_name}' 返回元组，提取第二个元素作为视频数据。")
            varr = varr[1] 
        # 2. 关键修正：初始化/更新总帧数和滑块
        # 检查 varr 是否存在，并且是否有 'frame' 维度
        if varr is not None and hasattr(varr, 'dims') and 'frame' in varr.dims:
            # 从 xarray.DataArray 的 .sizes 属性中获取总帧数
            new_total_frames = varr.sizes['frame']
            
            # 仅在总帧数首次设置或改变时更新 UI
            if new_total_frames != self.total_frames:
                self.total_frames = new_total_frames
                self.slider.setMaximum(self.total_frames - 1)
                
                # 确保当前帧索引不超过新的总帧数
                if self.current_frame >= self.total_frames:
                    self.current_frame = 0
                
                self.slider.setValue(self.current_frame)
                self.slider.setEnabled(True)
                
        else:
            # 对于非帧依赖的步骤 (如 'curve', 'none', 或 varr 为 None)
            self.total_frames = 1
            self.current_frame = 0
            self.slider.setMaximum(0) # 最大索引为 0
            self.slider.setValue(0)
            self.slider.setEnabled(False)

        # 3. 调用刷新帧函数
        # _update_visualization_frame 会根据 self.current_frame 和 self.total_frames 刷新显示和标签
        self._update_visualization_frame()
        
    def _update_visualization_frame(self):
        """根据当前的 self.current_frame 和 self.current_step_name 刷新显示。"""
        step_name = self.current_step_name
        vis_type = self.steps_map[step_name][2] # 保持 [3] 索引不变，假设您已修正 steps_map 的创建逻辑
        result = self.steps_results.get(step_name)

        if result is None: return

        frame_idx = self.current_frame
        image_array: Optional[np.ndarray] = None
        
        self.log_output.append(f"可视化类型: {vis_type}")
        self.log_output.append(f"结果: {result}")
        # 确保所有步骤都能获取到 video_data
        # varr: 用于作为背景的视频数组

        varr = self.processor.get_varr_for_vis(step_name) 
            
        # self.log_output.append(f"背景视频: {varr}")

        if vis_type != "video":
            self.total_frames = 1
            self.current_frame = 0
            self.slider.setMaximum(0) # 最大索引为 0
            self.slider.setValue(0)
            self.slider.setEnabled(False)

        try:
            if vis_type == "video":
                # 步骤 1, 2, 3, 4 (result 是 bool/None，应该使用 varr)
                # 🔴 修正点: 将 result 替换为 varr
                image_array = get_normalized_video_frame(varr, frame_idx) 

            elif vis_type == "split_video":

                # 步骤 5: Motion Correction (result 是 (varr_before, varr_after) tuple)
                varr_before, varr_after = varr
                image_array = create_mc_max_projection_comparison(varr_before, varr_after)
                self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}' (静态图): 运动校正前后最大投影对比")
            elif vis_type == "seeds":
                step_name = self.current_step_name 

                # 1. (通用) 从数据仓库加载最大投影图
                # (PNR 步骤重用 'seeds_init' 步骤生成的最大投影)
                max_proj = self.processor._load_data_from_repo('max_proj_seeds')
                
                if max_proj is None:
                    self.log_output.append(f"❌ 警告: 步骤 '{step_name}' 可视化失败：缺少 'max_proj_seeds' 背景图。")
                    # self._display_error_message("可视化失败: 缺少最大投影或种子数据。")
                    return
                
                # 2. 🔴 根据当前步骤执行分支逻辑 🔴
                seeds_to_keep = None
                seeds_to_remove = None

                if step_name == 'seeds_init':
                    # 案例 1: 'seeds_init' 步骤
                    # (只加载 'varr_seeds'，全部显示为白色)
                    seeds_to_keep = self.processor._load_data_from_repo('varr_seeds')
                    if seeds_to_keep is None:
                            raise ValueError("未找到 'varr_seeds' 数据。")
                    
                    self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}': 叠加所有初始种子 (白色)")

                elif step_name == 'peak_noise_ratio_refine':
                    # 案例 2: 'peak_noise_ratio_refine' 步骤
                    # (加载 'kept' 和 'removed' 两组)
                    seeds_to_keep = self.processor._load_data_from_repo('seeds_pnr_kept')
                    seeds_to_remove = self.processor._load_data_from_repo('seeds_pnr_removed')
                    
                    if seeds_to_keep is None:
                        raise ValueError("未找到 'seeds_pnr_kept' (保留的种子) 数据。")
                    
                    # seeds_to_remove 是可选的 (可能没有被移除的)
                    if seeds_to_remove is None:
                        self.log_output.append("-> PNR 可视化: 未找到 'seeds_pnr_removed' (移除的种子)。")

                    self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}': PNR 筛选 (白=保留, 红=移除)")
                    
                elif step_name == 'ks_refine':
                        # 案例 3: 'ks_refine' 步骤
                        seeds_to_keep = self.processor._load_data_from_repo('seeds_ks_kept')
                        seeds_to_remove = self.processor._load_data_from_repo('seeds_ks_removed')
                        
                        if seeds_to_keep is None:
                            raise ValueError("未找到 'seeds_ks_kept' (保留的种子) 数据。")
                        
                        if seeds_to_remove is None:
                            self.log_output.append("-> KS 可视化: 未找到 'seeds_ks_removed' (移除的种子)。")

                        self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}': KS 筛选 (白=保留, 红=移除)")
                    
                elif step_name == 'merge_seeds':
                        # 案例 4: 'merge_seeds' 步骤
                        seeds_to_keep = self.processor._load_data_from_repo('seeds_merged_kept')
                        seeds_to_remove = self.processor._load_data_from_repo('seeds_merged_removed')
                        
                        if seeds_to_keep is None:
                            raise ValueError("未找到 'seeds_merged_kept' (保留的种子) 数据。")
                        
                        if seeds_to_remove is None:
                            self.log_output.append("-> 合并可视化: 未找到 'seeds_merged_removed' (移除的种子)。")

                        self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}': 合并种子 (白=保留, 红=移除)")         
                
                image_array = create_seeds_visualization(
                    max_proj, 
                    seeds_to_keep, 
                    seeds_removed=seeds_to_remove
                )
                
            elif vis_type == "cnmf_init":
                # 步骤 11: CNMF A, C, b, f 初始化
                
                # 1. (静态) 加载所有计算好的组件
                A_init = self.processor._load_data_from_repo('A_init')
                C_init = self.processor._load_data_from_repo('C_init')
                b_init = self.processor._load_data_from_repo('b_init')
                f_init = self.processor._load_data_from_repo('f_init')
                
                # 2. 检查数据
                if A_init is None or C_init is None or b_init is None or f_init is None:
                    self.log_output.append(f"❌ 警告: 步骤 '{step_name}' 可视化失败：缺少 A, C, b 或 f 数据。")
                    return
                
                # 3. 调用新的可视化函数 (确保已从 visualization 导入)
                image_array = create_init_visualization_plot(
                    A_init, C_init, b_init, f_init
                )
                
                self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}': A, C, b, f 初始化 (2x2 面板)")
                    
                # else:
                #     self.log_output.append(f"-> 步骤 '{step_name}' 使用 'varr_seeds' 作为回退。")
                #     seeds_to_keep = self.processor._load_data_from_repo('varr_seeds')
                #     self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}': 叠加种子点 (基于最大投影)")


                #     # 3. 调用新的可视化函数
                #     # (兼容两种情况)
                #     image_array = create_seeds_visualization(
                #         max_proj, 
                #         seeds_to_keep, 
                #         seeds_removed=seeds_to_remove
                #     )


            elif vis_type == "curve_exploration":
                if isinstance(result, tuple) and len(result) >= 8:
                    status, sample_seeds, pnrs_mean, pnrs_values, signals_arr, noises_arr, noise_freq_list, fs = result

                    if status == "成功":
                        try:
                            # 可视化信噪比探索结果
                            image_array = create_pnr_refine_plot(
                                signals_arr=signals_arr,
                                noises_arr=noises_arr,
                                freq_list=noise_freq_list,
                                sample_seeds=sample_seeds,
                                fs=fs
                            )
                            
                            # self.display_image(image_array)
                        except Exception as e:
                            print(f"signal_arr: {signals_arr}, \n noises_arr: {noises_arr},\n freq_list: {noise_freq_list}")
                            print(f"signal_arr: {signals_arr.shape}, \n noises_arr: {noises_arr.shape},\n freq_list: {len(noise_freq_list)}")
                            print(f"❌ 信噪比可视化失败: {str(e)}")
                            self.log_output.append(f"❌ PNR可视化失败: {str(e)}")
                    else:
                        self.log_output.append("❌ 噪声频率探索步骤失败，无法生成曲线。")
                else:
                    self.log_output.append(f"❌ PNR可视化失败：返回数据结构不符合预期。")


                
            elif vis_type == "exploration":
                # 步骤 11, 13 (result 是 A_list)
                # Exploration plots are static and don't change per frame
                if frame_idx == 0:
                     # 假设 result 包含 A_list 和 penalties list
                     # TODO: 实际需要从 self.processor 获取探索参数和结果
                     A_list = self.processor.get_exploration_A_list(step_name) 
                     penalties = self.processor.get_exploration_penalties(step_name)
                     image_array = create_exploration_plot(varr, A_list, penalties, frame_idx)

            elif vis_type == "cnmf_update":
                # 步骤 12, 14, 15, 16 (result 是 (A, C, S) tuple)
                # TODO: 需要在 UI 上添加 Unit ID 选择框，这里假设 unit_id=0
                A_comp, C_comp, S_comp = result 
                unit_id = 0 # 假设默认显示第一个单元
                image_array = create_cnmf_update_plot(varr, A_comp, C_comp, S_comp, unit_id, frame_idx)

            elif vis_type == "none":
                self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}' (数据保存) 无可视化结果。")
                return

            if image_array is not None:
                # 确保图像是 RGB 或 BGR (H, W, 3) 格式
                h, w, c = image_array.shape
                bytes_per_line = c * w
                
                # 图像可能是 BGR 或 RGB，这里假设所有可视化函数返回 BGR (OpenCV 标准)
                q_image = QImage(image_array.data, w, h, bytes_per_line, QImage.Format_BGR888) 
                pixmap = QPixmap.fromImage(q_image)
                
                # 缩放以适应 QLabel
                pixmap = pixmap.scaled(self.vis_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.vis_label.setPixmap(pixmap)

            self.frame_label.setText(f"帧: {frame_idx + 1} / {self.total_frames}")
            self.slider.setValue(frame_idx)

        except Exception as e:
            error_trace = traceback.format_exc()
            self.vis_label.setText(f"可视化错误: {type(e).__name__}\n请检查日志")
            self.log_output.append(f"*** 可视化失败: {step_name} ***\n{error_trace}")
            self.slider.setEnabled(False)
    def toggle_playback(self):
        """开始/暂停视频播放，自动根据视频 FPS 设置播放速度。"""
        from PyQt5.QtCore import QTimer
        
        # 确保 QTimer 对象已初始化
        if not hasattr(self, 'visualization_timer') or self.visualization_timer is None:
            self.visualization_timer = QTimer(self)
            self.visualization_timer.timeout.connect(self._next_frame)
            
        if self.visualization_timer.isActive():
            self.stop_playback()
        else:
            # === 🔴 关键修正：获取并计算精确间隔 ===
            current_fps = 20.0 # 默认值，根据您的 ffmpeg 输出
            if hasattr(self, 'processor'):
                try:
                    # 尝试从处理器获取真实的 FPS
                    current_fps = self.processor.get_video_fps()
                except AttributeError:
                    # 如果方法不存在，使用默认值
                    pass 
            
            # 播放间隔 (ms) = 1000 / FPS
            interval_ms = max(1, int(1000 / current_fps)) # 确保间隔至少为 1ms
            
            self.visualization_timer.start(interval_ms) 
            self.play_pause_btn.setText("暂停 (⏸)")
            self.log_output.append(f"开始播放 (FPS: {current_fps:.2f}, 间隔: {interval_ms}ms)...")

    def stop_playback(self):
        """停止视频播放。"""
        if hasattr(self, 'visualization_timer') and self.visualization_timer is not None and self.visualization_timer.isActive():
            self.visualization_timer.stop()
            self.play_pause_btn.setText("开始 (▶)")
            self.log_output.append("播放暂停。")

    def _next_frame(self):
        """播放到下一帧。"""
        # 确保 total_frames 已被正确设置
        if self.total_frames is None:
            self.stop_playback()
            self.log_output.append("错误：总帧数(total_frames)未设置，无法播放。")
            return

        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._update_visualization_frame()
            # 自动更新滑块位置
            self.slider.setValue(self.current_frame) 
        else:
            # 播放到末尾后停止
            self.current_frame = 0
            self._update_visualization_frame()
            self.slider.setValue(self.current_frame)
            self.stop_playback() 
            self.log_output.append("播放完成。")

    def update_frame_from_slider(self, value: int):
        """拖动进度条跳转帧。（略微修正以保证同步）"""
        # 只有在值发生变化时才执行操作，避免信号重复触发
        if self.current_frame != value:
            self.current_frame = value
            self.stop_playback()
            self._update_visualization_frame()
        
    def seek_frame(self, offset: int):
        """快进/快退指定帧数。（略微修正以保证同步）"""
        self.stop_playback() # 跳转时停止播放
        
        # 确保 total_frames 已被设置
        if self.total_frames is None:
            self.log_output.append("错误：总帧数(total_frames)未设置，无法快进/退。")
            return
            
        new_frame = self.current_frame + offset
        # 确保帧数在有效范围内 [0, total_frames - 1]
        new_frame = max(0, min(self.total_frames - 1, new_frame))
        self.current_frame = new_frame
        
        # 通过设置滑块值来更新 UI 和可视化（假设滑块连接了 self.update_frame_from_slider）
        self.slider.setValue(new_frame)
        
    # =========================================================================
    # 8. 窗口关闭事件
    # =========================================================================
    
    def closeEvent(self, event):
        """关闭窗口时停止后台线程。"""
        if self.worker_thread.isRunning():
            self.worker_thread.terminate() # 强制终止 Dask 进程可能导致数据损坏，但对于 GUI 退出是必要的
            self.worker_thread.wait()
            self.log_output.append("后台计算线程已终止。")
        
        # TODO: 关闭 Dask Client/Cluster
        
        event.accept()