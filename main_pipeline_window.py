import os
import json
import sys
import traceback
from typing import Dict, Any, Optional, List, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, 
    QComboBox, QMessageBox, QGroupBox, QSpinBox, QDoubleSpinBox, QSlider, 
    QTextEdit, QFormLayout, QGridLayout, QScrollArea,QRadioButton, QButtonGroup,
    QSizePolicy, QFileDialog, QListWidget, QListWidgetItem, QAbstractItemView, QToolTip
)
from PyQt5 import QtWidgets
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QMutex, QLocale, QObject, QEvent, QTimer
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
    create_seeds_visualization, 
    create_pnr_refine_plot, 
    create_exploration_plot,   # 可以保留（其他步骤可能还用）
    create_cnmf_update_plot, 
    normalize_frame,
    create_mc_max_projection_comparison,
    create_init_visualization_plot,

    # ✅ 新增这两个
    create_spatial_exploration_plot,
    create_spatial_exploration_compare_plot,
    create_temporal_exploration_plot,
    create_temporal_exploration_compare_plot,
    create_save_data_dashboard,
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
    (16, "数据保存", "save_data", "none"),
]

SPATIAL_EXPLORE_STEPS = {"first_spatial_update_explore"}
TEMPORAL_EXPLORE_STEPS = {"first_temporal_update_explore"}
INTERACTIVE_EXPLORE_STEPS = SPATIAL_EXPLORE_STEPS | TEMPORAL_EXPLORE_STEPS
TEMPORAL_UPDATE_STEPS = {"first_temporal_update_exec"}
SAVE_DATA_MATRIX_ORDER = ["A", "C", "S", "YrA", "c0", "b0", "b", "f"]

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
    step_completed = pyqtSignal(str, object) # (步骤代码名, 结果数据) - 仅成功完成后触发
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
        self._current_task: Optional[Tuple[str, bool, Optional[str]]] = None # (步骤代码名, 是否为运行全部, 运行到步骤)
        self.all_steps_list = [name for _, _, name, _ in PIPELINE_STEPS]
        self.exploration_mode = "single"
        self.exploration_selected_penalty = None
        self.exploration_left_penalty = None
        self.exploration_right_penalty = None

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

            if result is False or result is None:
                self.signals.status_update.emit(step_name, "错误")
                self.signals.log_message.emit(f"--- 步骤 {step_name} 运行失败 ---")
            else:
                self.signals.status_update.emit(step_name, "已完成")
                # 仅在后台计算成功完成后，再通知主线程刷新界面
                self.signals.step_completed.emit(step_name, result)
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
        # 需要跳过的步骤
        SKIP_STEPS = {"noise_freq_exploration", "first_spatial_update_explore", "first_temporal_update_explore"}
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
                # 跳过探索类和噪声频率探索步骤
                if step_name in SKIP_STEPS:
                    self.signals.log_message.emit(f"[自动跳过] 步骤 {step_name} 属于参数探索/噪声频率探索，已跳过。")
                    continue
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
        self._is_updating_visualization = False
        self._last_vis_error_signature = None
        self._last_image_array: Optional[np.ndarray] = None
        self._save_data_hover_enabled = False
        self._save_data_hover_unit_map: Optional[np.ndarray] = None
        self._save_data_hover_left_width = 0
        self._save_data_dashboard_cache: Optional[Dict[str, Any]] = None
        self._save_data_dashboard_cache_key: Optional[Tuple[Any, ...]] = None
        
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
        self.vis_label.setMinimumSize(900, 560)
        self.vis_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vis_label.setMouseTracking(True)
        self.vis_label.installEventFilter(self)
        self.vis_label.setStyleSheet(
            """
            QLabel {
                border: 1px solid #3B4252;
                border-radius: 8px;
                background-color: #0F1117;
                color: #C9D1D9;
                padding: 4px;
            }
            """
        )
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
        self.explore_group = QGroupBox("探索结果切换")
        self.explore_group.setVisible(False)
        explore_layout = QGridLayout(self.explore_group)

        self.radio_single = QRadioButton("单参数查看")
        self.radio_compare = QRadioButton("双参数对比")
        self.radio_single.setChecked(True)

        self.explore_mode_group = QButtonGroup(self)
        self.explore_mode_group.addButton(self.radio_single)
        self.explore_mode_group.addButton(self.radio_compare)

        self.penalty_select_combo = QComboBox()
        self.left_penalty_combo = QComboBox()
        self.right_penalty_combo = QComboBox()
        self.btn_penalty_prev = QPushButton("上一个参数")
        self.btn_penalty_next = QPushButton("下一个参数")
        self.btn_compare_apply = QPushButton("应用双参数对比")

        self.radio_single.toggled.connect(self.on_exploration_control_changed)
        self.radio_compare.toggled.connect(self.on_exploration_control_changed)
        self.penalty_select_combo.currentIndexChanged.connect(self.on_exploration_control_changed)
        self.left_penalty_combo.currentIndexChanged.connect(self.on_exploration_control_changed)
        self.right_penalty_combo.currentIndexChanged.connect(self.on_exploration_control_changed)
        self.btn_penalty_prev.clicked.connect(lambda: self.switch_exploration_penalty(-1))
        self.btn_penalty_next.clicked.connect(lambda: self.switch_exploration_penalty(1))
        self.btn_compare_apply.clicked.connect(self.apply_exploration_compare_mode)

        explore_layout.addWidget(self.radio_single, 0, 0)
        explore_layout.addWidget(self.radio_compare, 0, 1)

        explore_layout.addWidget(QLabel("当前参数"), 1, 0)
        explore_layout.addWidget(self.penalty_select_combo, 1, 1)
        explore_layout.addWidget(self.btn_penalty_prev, 1, 2)
        explore_layout.addWidget(self.btn_penalty_next, 1, 3)

        explore_layout.addWidget(QLabel("左侧参数"), 2, 0)
        explore_layout.addWidget(self.left_penalty_combo, 2, 1)

        explore_layout.addWidget(QLabel("右侧参数"), 3, 0)
        explore_layout.addWidget(self.right_penalty_combo, 3, 1)
        explore_layout.addWidget(self.btn_compare_apply, 3, 2, 1, 2)

        vis_layout.addWidget(self.explore_group)

        self.temporal_view_group = QGroupBox("Temporal 结果切换")
        self.temporal_view_group.setVisible(False)
        temporal_view_layout = QHBoxLayout(self.temporal_view_group)
        temporal_view_layout.addWidget(QLabel("显示内容"))
        self.temporal_view_combo = QComboBox()
        self.temporal_view_combo.addItems(["update", "merge"])
        self.temporal_view_combo.currentIndexChanged.connect(self.on_temporal_update_view_changed)
        temporal_view_layout.addWidget(self.temporal_view_combo)
        vis_layout.addWidget(self.temporal_view_group)

        self.save_data_group = QGroupBox("数据保存设置与单元筛选")
        self.save_data_group.setVisible(False)
        save_layout = QGridLayout(self.save_data_group)

        save_layout.addWidget(QLabel("保存矩阵(多选)"), 0, 0)
        self.save_matrix_list = QListWidget()
        self.save_matrix_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.save_matrix_list.setMaximumHeight(120)
        self.save_matrix_list.itemChanged.connect(self.on_save_data_controls_changed)
        save_layout.addWidget(self.save_matrix_list, 1, 0, 3, 1)

        save_layout.addWidget(QLabel("保存格式"), 0, 1)
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["zarr", "netcdf", "csv", "npy"])
        self.save_format_combo.currentIndexChanged.connect(self.on_save_data_controls_changed)
        save_layout.addWidget(self.save_format_combo, 1, 1)

        save_layout.addWidget(QLabel("保存目录"), 2, 1)
        self.save_output_dir_edit = QLineEdit()
        self.save_output_dir_edit.editingFinished.connect(self.on_save_data_controls_changed)
        save_layout.addWidget(self.save_output_dir_edit, 3, 1)
        self.btn_browse_save_dir = QPushButton("选择目录")
        self.btn_browse_save_dir.clicked.connect(self.on_browse_save_output_dir)
        save_layout.addWidget(self.btn_browse_save_dir, 3, 2)

        save_layout.addWidget(QLabel("预览 unit_id"), 0, 2)
        self.save_unit_combo = QComboBox()
        self.save_unit_combo.currentIndexChanged.connect(self.on_save_data_unit_changed)
        save_layout.addWidget(self.save_unit_combo, 1, 2)

        self.btn_exclude_unit = QPushButton("排除当前 unit")
        self.btn_exclude_unit.clicked.connect(self.on_exclude_current_unit)
        save_layout.addWidget(self.btn_exclude_unit, 2, 2)

        self.btn_reset_excluded = QPushButton("清空排除列表")
        self.btn_reset_excluded.clicked.connect(self.on_reset_excluded_units)
        save_layout.addWidget(self.btn_reset_excluded, 2, 3)

        self.save_excluded_label = QLabel("已排除: 无")
        save_layout.addWidget(self.save_excluded_label, 3, 3)

        self.btn_save_data_now = QPushButton("立即保存（覆盖同名输出）")
        self.btn_save_data_now.setStyleSheet("background-color: #90caf9;")
        self.btn_save_data_now.clicked.connect(self.on_save_data_now_clicked)
        save_layout.addWidget(self.btn_save_data_now, 4, 0, 1, 2)

        self.save_data_status_label = QLabel("说明：再次点击会覆盖同名保存结果。")
        save_layout.addWidget(self.save_data_status_label, 4, 2, 1, 2)

        vis_layout.addWidget(self.save_data_group)

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
        self.worker_signals.step_completed.connect(self.handle_step_completed)
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
            current_status = self.steps_status.get(current_step)

            # 关键修复：自动运行模式下，若当前步骤失败则立即中止，避免死循环反复重试
            if current_status == "错误":
                QMessageBox.critical(
                    self,
                    "自动运行已中止",
                    f"步骤 '{self.steps_map[current_step][1]}' 运行失败，已停止自动运行。\n请修正参数或数据后手动重试。"
                )
                delattr(self, '_auto_run_target')
                return
            
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
    def refresh_exploration_controls(self, step_name: str):
        data = self.processor.get_exploration_result(step_name)
        is_explore = step_name in INTERACTIVE_EXPLORE_STEPS and data is not None

        self.explore_group.setVisible(is_explore)
        if not is_explore:
            return
        if data is None:
            return

        penalties = data.get("penalty_list", [])

        # 避免每次刷新都重置用户选择
        existing_items = [self.penalty_select_combo.itemText(i) for i in range(self.penalty_select_combo.count())]
        new_items = [str(p) for p in penalties]
        should_rebuild = existing_items != new_items

        if should_rebuild:
            for combo in [self.penalty_select_combo, self.left_penalty_combo, self.right_penalty_combo]:
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(new_items)
                combo.blockSignals(False)

        state = self.processor.get_exploration_state(step_name)
        default_penalty = state.get("selected_penalty", data.get("default_penalty", penalties[0] if penalties else None))

        if default_penalty is not None and self.penalty_select_combo.count() > 0:
            idx = self.penalty_select_combo.findText(str(default_penalty))
            if idx >= 0 and self.penalty_select_combo.currentIndex() != idx:
                self.penalty_select_combo.setCurrentIndex(idx)

        if len(penalties) >= 2 and should_rebuild:
            li = self.left_penalty_combo.findText(str(penalties[0]))
            ri = self.right_penalty_combo.findText(str(penalties[1]))
            if li >= 0:
                self.left_penalty_combo.setCurrentIndex(li)
            if ri >= 0:
                self.right_penalty_combo.setCurrentIndex(ri)

    def switch_exploration_penalty(self, direction: int):
        step_name = self.current_step_name
        if step_name not in INTERACTIVE_EXPLORE_STEPS or self.penalty_select_combo.count() == 0:
            return
        cur = self.penalty_select_combo.currentIndex()
        if cur < 0:
            cur = 0
        nxt = max(0, min(self.penalty_select_combo.count() - 1, cur + direction))
        if nxt != cur:
            self.penalty_select_combo.setCurrentIndex(nxt)

    def apply_exploration_compare_mode(self):
        step_name = self.current_step_name
        if step_name not in INTERACTIVE_EXPLORE_STEPS:
            return
        if self.left_penalty_combo.count() == 0 or self.right_penalty_combo.count() == 0:
            return
        self.radio_compare.setChecked(True)
        self.on_exploration_control_changed()
                
    def on_exploration_control_changed(self):
        step_name = self.current_step_name
        if step_name not in INTERACTIVE_EXPLORE_STEPS:
            return

        state = {
            "mode": "compare" if self.radio_compare.isChecked() else "single",
            "selected_penalty": self.penalty_select_combo.currentText(),
            "left_penalty": self.left_penalty_combo.currentText(),
            "right_penalty": self.right_penalty_combo.currentText(),
        }
        self.processor.set_exploration_state(step_name, state)
        self._update_visualization_frame()

    def _pick_penalty_result(self, data: dict, penalty_text: str):
        """容错选择 penalty 结果，避免浮点字符串精度导致 key 命中失败。"""
        results = data.get("results", {})
        if not results:
            return 0.0, None

        keys = sorted([float(k) for k in results.keys()])
        try:
            target = float(penalty_text)
            best = min(keys, key=lambda x: abs(x - target))
        except Exception:
            best = keys[0]
        return float(best), results.get(best)

    def refresh_temporal_update_controls(self, step_name: str):
        is_temporal_update = step_name in TEMPORAL_UPDATE_STEPS
        self.temporal_view_group.setVisible(is_temporal_update)
        if not is_temporal_update:
            return

        # 兼容旧键（例如 first_temporal_update_*）
        legacy_step = step_name.replace("_exec", "")
        has_update = (
            self.processor._load_data_from_repo(f"{step_name}_update_vis_array") is not None
            or self.processor._load_data_from_repo(f"{legacy_step}_update_vis_array") is not None
            or self.processor._load_data_from_repo(f"{step_name}_c_s_vis_array") is not None
            or self.processor._load_data_from_repo(f"{legacy_step}_c_s_vis_array") is not None
        )
        has_merge = (
            self.processor._load_data_from_repo(f"{step_name}_merge_vis_array") is not None
            or self.processor._load_data_from_repo(f"{legacy_step}_merge_vis_array") is not None
        )

        current = self.temporal_view_combo.currentText()
        target_items = []
        if has_update:
            target_items.append("update")
        if has_merge:
            target_items.append("merge")
        if not target_items:
            target_items = ["update"]

        existing_items = [self.temporal_view_combo.itemText(i) for i in range(self.temporal_view_combo.count())]
        should_rebuild = existing_items != target_items

        if should_rebuild:
            self.temporal_view_combo.blockSignals(True)
            self.temporal_view_combo.clear()
            self.temporal_view_combo.addItems(target_items)
            self.temporal_view_combo.blockSignals(False)

        # 保持用户选择，不再每次刷新都回到 update
        if current and self.temporal_view_combo.findText(current) >= 0:
            idx = self.temporal_view_combo.findText(current)
        else:
            idx = 0
        if self.temporal_view_combo.currentIndex() != idx:
            self.temporal_view_combo.setCurrentIndex(idx)

    def on_temporal_update_view_changed(self):
        if self.current_step_name not in TEMPORAL_UPDATE_STEPS:
            return
        self._update_visualization_frame()

    def refresh_save_data_controls(self, step_name: str):
        is_save_data = step_name == "save_data"
        self.save_data_group.setVisible(is_save_data)
        if not is_save_data:
            self._save_data_hover_enabled = False
            self._save_data_hover_unit_map = None
            self._save_data_hover_left_width = 0
            self._save_data_dashboard_cache = None
            self._save_data_dashboard_cache_key = None
            QToolTip.hideText()
            return

        params = self.processor.get_step_params("save_data") or {}
        selected = params.get("selected_matrices", ["A", "C", "S", "YrA", "c0", "b0", "b", "f"])
        if not isinstance(selected, list):
            selected = ["A", "C", "S", "c0", "b0", "b", "f"]
        selected_set = set(str(x) for x in selected)

        available = self.processor._resolve_save_data_inputs()

        self.save_matrix_list.blockSignals(True)
        self.save_matrix_list.clear()
        for name in SAVE_DATA_MATRIX_ORDER:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if name in selected_set else Qt.Unchecked)
            self.save_matrix_list.addItem(item)
        self.save_matrix_list.blockSignals(False)

        save_format = str(params.get("save_format", "zarr")).lower()
        idx = self.save_format_combo.findText(save_format)
        if idx < 0:
            idx = self.save_format_combo.findText("zarr")
        if idx >= 0 and self.save_format_combo.currentIndex() != idx:
            self.save_format_combo.blockSignals(True)
            self.save_format_combo.setCurrentIndex(idx)
            self.save_format_combo.blockSignals(False)

        output_dir = str(params.get("output_dir", "./minian_output"))
        if self.save_output_dir_edit.text() != output_dir:
            self.save_output_dir_edit.setText(output_dir)

        excluded_raw = params.get("excluded_unit_ids", [])
        excluded = []
        if isinstance(excluded_raw, list):
            for x in excluded_raw:
                try:
                    excluded.append(int(x))
                except Exception:
                    continue

        # 保留用户当前选择，避免每次刷新被重置到首项
        prev_unit_text = self.save_unit_combo.currentText().strip() if self.save_unit_combo.count() > 0 else ""

        unit_ids = []
        unit_pool = set()
        a_data = available.get("A")
        if a_data is not None and "unit_id" in a_data.coords:
            for u in a_data.coords["unit_id"].values:
                unit_pool.add(int(u))
        c_data = available.get("C")
        c_units = []
        if c_data is not None and "unit_id" in c_data.coords:
            c_units = [int(u) for u in c_data.coords["unit_id"].values]

        # 优先使用 C 的 unit 列表，保证右侧 C 曲线可用；若 C 缺失再回退 A
        if len(c_units) > 0:
            base_units = sorted(set(c_units))
        else:
            base_units = sorted(unit_pool)

        for uid in base_units:
            if uid not in set(excluded):
                unit_ids.append(uid)

        self.save_unit_combo.blockSignals(True)
        self.save_unit_combo.clear()
        if len(unit_ids) == 0:
            self.save_unit_combo.addItem("(无可用 unit)")
            self.save_unit_combo.setEnabled(False)
            self.btn_exclude_unit.setEnabled(False)
        else:
            for uid in unit_ids:
                self.save_unit_combo.addItem(str(uid))
            self.save_unit_combo.setEnabled(True)
            self.btn_exclude_unit.setEnabled(True)

            # 还原此前选择
            if prev_unit_text:
                idx_prev = self.save_unit_combo.findText(prev_unit_text)
                if idx_prev >= 0:
                    self.save_unit_combo.setCurrentIndex(idx_prev)
                else:
                    self.save_unit_combo.setCurrentIndex(0)
            else:
                self.save_unit_combo.setCurrentIndex(0)
        self.save_unit_combo.blockSignals(False)

        ex_text = "无" if len(excluded) == 0 else ", ".join(str(x) for x in excluded[:12])
        if len(excluded) > 12:
            ex_text += " ..."
        self.save_excluded_label.setText(f"已排除: {ex_text}")

        a_obj = available.get("A")
        c_obj = available.get("C")
        a_sig = tuple(a_obj.sizes.items()) if isinstance(a_obj, xr.DataArray) else ()
        c_sig = tuple(c_obj.sizes.items()) if isinstance(c_obj, xr.DataArray) else ()
        new_cache_key = (id(a_obj), id(c_obj), a_sig, c_sig)
        if self._save_data_dashboard_cache_key != new_cache_key:
            self._save_data_dashboard_cache = None
            self._save_data_dashboard_cache_key = new_cache_key

    def _collect_save_data_ui_state(self) -> Dict[str, Any]:
        selected = []
        for i in range(self.save_matrix_list.count()):
            item = self.save_matrix_list.item(i)
            if item is not None and item.checkState() == Qt.Checked:
                selected.append(item.text())

        params = self.processor.get_step_params("save_data") or {}
        old_excluded = params.get("excluded_unit_ids", [])
        excluded = []
        if isinstance(old_excluded, list):
            for x in old_excluded:
                try:
                    excluded.append(int(x))
                except Exception:
                    continue

        return {
            "selected_matrices": selected,
            "save_format": self.save_format_combo.currentText().strip().lower(),
            "output_dir": self.save_output_dir_edit.text().strip() or "./minian_output",
            "excluded_unit_ids": excluded,
        }

    def on_save_data_controls_changed(self):
        if self.current_step_name != "save_data":
            return
        new_state = self._collect_save_data_ui_state()
        self.processor.update_params("save_data", new_state)
        self._update_visualization_frame()

    def on_browse_save_output_dir(self):
        cur = self.save_output_dir_edit.text().strip() or self.processor.video_folder
        picked = QFileDialog.getExistingDirectory(self, "选择保存目录", cur)
        if picked:
            self.save_output_dir_edit.setText(picked)
            self.on_save_data_controls_changed()

    def on_save_data_unit_changed(self):
        if self.current_step_name != "save_data":
            return
        self._update_visualization_frame()

    def on_exclude_current_unit(self):
        if self.current_step_name != "save_data":
            return
        if not self.save_unit_combo.isEnabled() or self.save_unit_combo.count() == 0:
            return
        txt = self.save_unit_combo.currentText().strip()
        try:
            uid = int(txt)
        except Exception:
            return

        params = self.processor.get_step_params("save_data") or {}
        excluded_raw = params.get("excluded_unit_ids", [])
        excluded = []
        if isinstance(excluded_raw, list):
            for x in excluded_raw:
                try:
                    excluded.append(int(x))
                except Exception:
                    continue
        if uid not in excluded:
            excluded.append(uid)
        params["excluded_unit_ids"] = sorted(set(excluded))
        self.processor.update_params("save_data", params)
        self.refresh_save_data_controls("save_data")
        self._update_visualization_frame()

    def on_reset_excluded_units(self):
        if self.current_step_name != "save_data":
            return
        params = self.processor.get_step_params("save_data") or {}
        params["excluded_unit_ids"] = []
        self.processor.update_params("save_data", params)
        self.refresh_save_data_controls("save_data")
        self._update_visualization_frame()

    def on_save_data_now_clicked(self):
        if self.current_step_name != "save_data":
            self.current_step_name = "save_data"
            self.update_parameters_panel()
            self.update_step_list_widget(force_select=True)
            self.visualize_current_step()

        self.on_save_data_controls_changed()
        self.save_data_status_label.setText("正在保存...（同名文件将被覆盖）")
        self.run_current_step()
        
    def run_current_step(self):
        """
        运行当前选定步骤的逻辑。
        1. 检查参数更新并保存到配置文件。
        2. 标记后续步骤为 '未运行' 并清除缓存。
        3. 启动工作线程执行当前步骤。
        """
        step_name = self.current_step_name

        if step_name == "save_data":
            self.on_save_data_controls_changed()
        
        # 1. 检查并保存参数
        if step_name != "save_data" and self._check_and_save_parameters(step_name):
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

        # 关键修复：若当前步骤已处于错误态，直接停止，不再重复触发同一步骤
        if current_status == "错误":
            self.log_output.append(f"自动运行中止: 步骤 {current_step} 运行失败，不再自动重试。")
            self._set_ui_running_state(False)
            if hasattr(self, '_auto_run_target'):
                delattr(self, '_auto_run_target')
            return
        
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
        """处理步骤运行结果，仅存储数据，不立即重绘。"""
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

    def handle_step_completed(self, step_name: str, result: Any):
        """后台步骤完成后触发：通过事件队列延迟刷新，避免界面阻塞。"""
        # 结果兜底保存
        self.steps_results[step_name] = result

        # “运行到指定步骤”模式：每完成一步都刷新该步骤可视化
        force_switch = hasattr(self, '_auto_run_target')

        # 通过 queued 调度在主线程空闲时刷新，降低卡顿风险
        QTimer.singleShot(0, lambda sn=step_name, fs=force_switch: self._refresh_after_step_completed(sn, fs))

    def _refresh_after_step_completed(self, step_name: str, force_switch: bool = False):
        if self.steps_status.get(step_name) != "已完成":
            return

        if step_name == "save_data" and hasattr(self, "save_data_status_label"):
            self.save_data_status_label.setText("保存完成。再次点击会覆盖同名保存结果。")

        prev_step = self.current_step_name
        try:
            if force_switch:
                self.current_step_name = step_name
                self.update_parameters_panel()
                self.update_step_list_widget(force_select=True)
                self.visualize_current_step()
            else:
                if step_name != self.current_step_name:
                    return
                self.visualize_current_step()
                self.update_step_list_widget(force_select=True)
        finally:
            # “运行到指定步骤”过程中，保持当前步骤为已完成步骤，
            # 便于用户看到逐步更新的可视化。
            if not force_switch:
                self.current_step_name = prev_step
            
    def visualize_current_step(self):
        """
        根据当前选定的步骤，获取相应的可视化数据，并设置总帧数和滑块。
        最后调用 _update_visualization_frame 刷新显示。
        """
        step_name = self.current_step_name
        self.refresh_temporal_update_controls(step_name)
        self.refresh_save_data_controls(step_name)
        # 🚨 警告修复点：获取步骤状态并检查是否已完成
        status = self.steps_status.get(step_name)
        
        if status != "已完成" and step_name != "save_data":
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
        if self._is_updating_visualization:
            return

        self._is_updating_visualization = True
        step_name = self.current_step_name
        vis_type = self.steps_map[step_name][2] # 保持 [3] 索引不变，假设您已修正 steps_map 的创建逻辑
        result = self.steps_results.get(step_name)

        if result is None and step_name != "save_data":
            self._is_updating_visualization = False
            return

        frame_idx = self.current_frame
        image_array: Optional[np.ndarray] = None
        
        self.log_output.append(f"可视化类型: {vis_type}")
        self.log_output.append(f"结果: {result}")
        # 确保所有步骤都能获取到 video_data
        # varr: 用于作为背景的视频数组

        varr = self.processor.get_varr_for_vis(step_name) 

        self.refresh_temporal_update_controls(step_name)
        self.refresh_save_data_controls(step_name)
            
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
                data = self.processor.get_exploration_result(step_name)
                if data is None:
                    self.vis_label.setText("尚无探索结果，请先运行该步骤。")
                    return

                self.refresh_exploration_controls(step_name)

                state = self.processor.get_exploration_state(step_name)
                mode = state.get("mode", "single")

                if mode == "compare":
                    left_pen, left_res = self._pick_penalty_result(
                        data,
                        state.get("left_penalty", str(data["penalty_list"][0]))
                    )
                    default_right = data["penalty_list"][1 if len(data['penalty_list']) > 1 else 0]
                    right_pen, right_res = self._pick_penalty_result(
                        data,
                        state.get("right_penalty", str(default_right))
                    )
                    if left_res is None or right_res is None:
                        self.vis_label.setText("参数对比数据不可用，请重新选择参数。")
                        return

                    if step_name in SPATIAL_EXPLORE_STEPS:
                        image_array = create_spatial_exploration_compare_plot(
                            left_res,
                            right_res,
                            float(left_pen),
                            float(right_pen)
                        )
                        self.vis_label.setText(
                            f"步骤 '{self.steps_map[step_name][1]}': spatial 左右对比 ({left_pen} vs {right_pen})"
                        )
                    else:
                        image_array = create_temporal_exploration_compare_plot(
                            left_res,
                            right_res,
                            float(left_pen),
                            float(right_pen)
                        )
                        self.vis_label.setText(
                            f"步骤 '{self.steps_map[step_name][1]}': temporal 上下对比 ({left_pen} vs {right_pen})"
                        )

                else:
                    selected_pen, cur_res = self._pick_penalty_result(
                        data,
                        state.get("selected_penalty", str(data.get("default_penalty", data["penalty_list"][0])))
                    )
                    if cur_res is None:
                        self.vis_label.setText("参数结果不可用，请重新选择参数。")
                        return

                    if step_name in SPATIAL_EXPLORE_STEPS:
                        image_array = create_spatial_exploration_plot(cur_res, float(selected_pen))
                    else:
                        image_array = create_temporal_exploration_plot(cur_res, float(selected_pen))
                    self.vis_label.setText(
                        f"步骤 '{self.steps_map[step_name][1]}': sparse_penalty = {selected_pen}"
                    )

            elif vis_type == "cnmf_update":
                # 空间更新执行步骤：显示整视野 2x2 结果，不再选单个 unit
                if step_name in {"first_spatial_update_exec"}:
                    image_array = self.processor._load_data_from_repo(f"{step_name}_vis_array")
                    if image_array is None:
                        self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}' 尚无整视野可视化结果。")
                        return
                elif step_name in TEMPORAL_UPDATE_STEPS:
                    view_mode = self.temporal_view_combo.currentText() if self.temporal_view_combo.count() > 0 else "update"
                    legacy_step = step_name.replace("_exec", "")
                    image_array = self.processor._load_data_from_repo(f"{step_name}_{view_mode}_vis_array")
                    if image_array is None:
                        image_array = self.processor._load_data_from_repo(f"{legacy_step}_{view_mode}_vis_array")
                    if image_array is None and view_mode == "update":
                        image_array = self.processor._load_data_from_repo(f"{step_name}_c_s_vis_array")
                    if image_array is None and view_mode == "update":
                        image_array = self.processor._load_data_from_repo(f"{legacy_step}_c_s_vis_array")
                    if image_array is None:
                        self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}' 的 {view_mode} 可视化结果不存在。")
                        return
                    self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}': temporal {view_mode} 热图")
                else:
                    # 其他 cnmf_update 维持原逻辑
                    if varr is None:
                        self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}' 缺少背景视频数据，无法可视化。")
                        return
                    A_comp, C_comp, S_comp = result
                    unit_id = 0
                    image_array = create_cnmf_update_plot(varr, A_comp, C_comp, S_comp, unit_id, frame_idx)

            elif vis_type == "none":
                if step_name == "save_data":
                    params = self.processor.get_step_params("save_data") or {}
                    excluded_raw = params.get("excluded_unit_ids", [])
                    excluded_units = []
                    if isinstance(excluded_raw, list):
                        for x in excluded_raw:
                            try:
                                excluded_units.append(int(x))
                            except Exception:
                                continue

                    available = self.processor._resolve_save_data_inputs()
                    A = available.get("A")
                    C = available.get("C")

                    unit_id = None
                    if self.save_unit_combo.isEnabled() and self.save_unit_combo.count() > 0:
                        try:
                            unit_id = int(self.save_unit_combo.currentText())
                        except Exception:
                            unit_id = None

                    # 若当前选择 unit 在 C 中不存在，自动回退到 C 的首个可用 unit，避免“无曲线”
                    if C is not None and "unit_id" in getattr(C, "coords", {}):
                        c_units = [int(u) for u in C.coords["unit_id"].values]
                        if len(c_units) > 0 and (unit_id is None or unit_id not in set(c_units)):
                            unit_id = int(c_units[0])
                            if self.save_unit_combo.isEnabled():
                                idx = self.save_unit_combo.findText(str(unit_id))
                                if idx >= 0 and self.save_unit_combo.currentIndex() != idx:
                                    self.save_unit_combo.blockSignals(True)
                                    self.save_unit_combo.setCurrentIndex(idx)
                                    self.save_unit_combo.blockSignals(False)

                    dashboard = create_save_data_dashboard(
                        A=A,
                        C=C,
                        unit_id=unit_id,
                        excluded_units=excluded_units,
                        dashboard_cache=self._save_data_dashboard_cache,
                    )
                    image_array = dashboard.get("image")
                    if isinstance(dashboard.get("cache"), dict):
                        self._save_data_dashboard_cache = dashboard.get("cache")
                    self._save_data_hover_unit_map = dashboard.get("unit_id_map")
                    self._save_data_hover_left_width = int(dashboard.get("left_width", 0) or 0)
                    self._save_data_hover_enabled = (
                        isinstance(self._save_data_hover_unit_map, np.ndarray)
                        and self._save_data_hover_unit_map.ndim == 2
                        and self._save_data_hover_left_width > 0
                    )
                    self.vis_label.setText(
                        f"步骤 '{self.steps_map[step_name][1]}': 可交互预览 (unit 空间分布 + 时间曲线)"
                    )
                else:
                    self.vis_label.setText(f"步骤 '{self.steps_map[step_name][1]}' (数据保存) 无可视化结果。")
                    return

            if image_array is not None:
                self._last_image_array = image_array
                self._render_image_array(image_array)

            self.frame_label.setText(f"帧: {frame_idx + 1} / {self.total_frames}")
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)

            # 可视化成功后清空错误签名，允许后续新错误正常上报
            self._last_vis_error_signature = None

        except Exception as e:
            error_trace = traceback.format_exc()
            self.vis_label.setText(f"可视化错误: {type(e).__name__}\n请检查日志")
            err_sig = (step_name, type(e).__name__, str(e))
            if err_sig != self._last_vis_error_signature:
                self.log_output.append(f"*** 可视化失败: {step_name} ***\n{error_trace}")
                self._last_vis_error_signature = err_sig
            self.slider.setEnabled(False)
            # 发生可视化异常时停止播放，避免定时器重复触发导致“死循环”
            self.stop_playback()
        finally:
            self._is_updating_visualization = False

    def _render_image_array(self, image_array: np.ndarray):
        """统一渲染入口：按可视区域填充并保持平滑缩放。"""
        if image_array is None:
            return
        if not isinstance(image_array, np.ndarray) or image_array.ndim != 3 or image_array.shape[2] != 3:
            self.vis_label.setText("可视化数据格式异常")
            return

        h, w, c = image_array.shape
        bytes_per_line = c * w
        q_image = QImage(image_array.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        # 使用 KeepAspectRatio 避免边缘被裁剪，保证整图完整可见
        pixmap = pixmap.scaled(self.vis_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.vis_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        """窗口尺寸变化时，重绘最后一帧/最后一张图，让画面持续填充。"""
        super().resizeEvent(event)
        if self._last_image_array is not None:
            self._render_image_array(self._last_image_array)

    def eventFilter(self, obj, event):
        if obj is self.vis_label:
            if event.type() == QEvent.MouseMove:
                self._handle_save_data_hover(event)
            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._handle_save_data_click(event)
        return super().eventFilter(obj, event)

    def _map_event_to_save_data_unit(self, event) -> Optional[int]:
        if self.current_step_name != "save_data" or not self._save_data_hover_enabled:
            return None
        if self.vis_label.pixmap() is None or self._last_image_array is None:
            return None

        pm = self.vis_label.pixmap()
        pm_w, pm_h = pm.width(), pm.height()
        if pm_w <= 0 or pm_h <= 0:
            return None

        lbl_w, lbl_h = self.vis_label.width(), self.vis_label.height()
        ox = (lbl_w - pm_w) // 2
        oy = (lbl_h - pm_h) // 2

        ex, ey = event.pos().x(), event.pos().y()
        if ex < ox or ey < oy or ex >= ox + pm_w or ey >= oy + pm_h:
            return None

        img_h, img_w = self._last_image_array.shape[:2]
        ix = int((ex - ox) * img_w / max(1, pm_w))
        iy = int((ey - oy) * img_h / max(1, pm_h))
        if ix < 0 or iy < 0 or ix >= img_w or iy >= img_h:
            return None

        if ix >= self._save_data_hover_left_width:
            return None

        m = self._save_data_hover_unit_map
        if m is None:
            return None

        mh, mw = m.shape
        mx = int(ix * mw / max(1, self._save_data_hover_left_width))
        my = int(iy * mh / max(1, img_h))
        if mx < 0 or my < 0 or mx >= mw or my >= mh:
            return None

        uid = int(m[my, mx])
        return uid if uid >= 0 else None

    def _handle_save_data_hover(self, event):
        uid = self._map_event_to_save_data_unit(event)
        if uid is None:
            QToolTip.hideText()
            return
        QToolTip.showText(event.globalPos(), f"unit_id: {uid}", self.vis_label)

    def _handle_save_data_click(self, event):
        uid = self._map_event_to_save_data_unit(event)
        if uid is None or not self.save_unit_combo.isEnabled():
            return
        idx = self.save_unit_combo.findText(str(uid))
        if idx >= 0:
            self.save_unit_combo.setCurrentIndex(idx)
            self._update_visualization_frame()

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