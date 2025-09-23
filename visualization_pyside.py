import matplotlib
matplotlib.use('Qt5Agg')


import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout, QPushButton, QSlider
from PyQt5.QtCore import Qt, QTimer


# 这是一个可以嵌入到 PyQt5 界面的 Matplotlib 图形画布
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(
            figsize=(width, height), dpi=dpi, constrained_layout=True
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_frame_on(False)
        self.fig.patch.set_facecolor('none')
        self.fig.tight_layout()
        super(MplCanvas, self).__init__(self.fig)


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 视频画布
        self.canvas = MplCanvas(self, width=6, height=6)
        self.layout.addWidget(self.canvas)

        self.current_frame_idx = 0
        self.varr = None
        self.im = None

        # 视频播放定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # 播放控制条布局
        self.controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶ 播放")
        self.pause_btn = QPushButton("⏸ 暂停")
        self.rewind_btn = QPushButton("快退")
        self.forward_btn = QPushButton("快进")
        self.progress_slider = QSlider(Qt.Horizontal)
        self.frame_label = QLabel("0 / 0")

        self.controls_layout.addWidget(self.rewind_btn)
        self.controls_layout.addWidget(self.pause_btn)
        self.controls_layout.addWidget(self.play_btn)
        self.controls_layout.addWidget(self.forward_btn)
        self.controls_layout.addWidget(self.progress_slider)
        self.controls_layout.addWidget(self.frame_label)

        self.layout.addLayout(self.controls_layout)

        # 连接信号与槽
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.rewind_btn.clicked.connect(self.rewind)
        self.forward_btn.clicked.connect(self.forward)
        
        # 修正: 将 progress_slider 的信号连接到新的 set_position 方法
        self.progress_slider.sliderMoved.connect(self.set_position)
        self.progress_slider.sliderReleased.connect(self.set_position)
        
        # 默认禁用控件，直到视频加载完成
        self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled):
        """启用或禁用播放控制按钮和进度条"""
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.rewind_btn.setEnabled(enabled)
        self.forward_btn.setEnabled(enabled)
        self.progress_slider.setEnabled(enabled)

    def load_video_data(self, varr: xr.DataArray):
        """加载视频数据"""
        # 强制Dask数组计算，将所有数据加载到内存中
        self.varr = varr.compute()
        self.current_frame_idx = 0
        
        # 更新进度条的最大值和当前位置
        self.progress_slider.setMaximum(self.varr.sizes['frame'] - 1)
        self.progress_slider.setValue(0)
        
        # 显示第一帧
        self.update_frame(0)
        
        # 启用控件
        self.set_controls_enabled(True)

    def update_frame(self, frame_idx: int):
        """更新显示的视频帧"""
        if self.varr is None or frame_idx < 0 or frame_idx >= self.varr.sizes['frame']:
            self.pause()
            return
        
        self.current_frame_idx = frame_idx
        
        frame_data = self.varr.isel(frame=frame_idx).compute()
        if self.im is None:
            self.im = self.canvas.axes.imshow(frame_data, cmap='gray')
            self.canvas.axes.set_title(f"帧: {frame_idx + 1}/{self.varr.sizes['frame']}")
            self.canvas.axes.set_xticks([])
            self.canvas.axes.set_yticks([])
            self.canvas.axes.set_frame_on(False)
        else:
            self.im.set_array(frame_data)
        
        self.canvas.draw()
        
        self.progress_slider.blockSignals(True) # 阻止信号以避免循环
        self.progress_slider.setValue(frame_idx)
        self.progress_slider.blockSignals(False)
        self.frame_label.setText(f"{frame_idx + 1} / {self.varr.sizes['frame']}")

    def next_frame(self):
        """更新到下一帧"""
        self.update_frame(self.current_frame_idx + 1)

    def play(self):
        """开始播放视频"""
        if self.varr is not None and not self.timer.isActive():
            self.timer.start(33) # 启动定时器，每 33 毫秒触发一次（大约30帧/秒）

    def pause(self):
        """暂停播放视频"""
        if self.timer.isActive():
            self.timer.stop()

    def rewind(self):
        """快退1秒（约30帧）"""
        self.update_frame(max(0, self.current_frame_idx - 30))

    def forward(self):
        """快进1秒（约30帧）"""
        self.update_frame(min(self.varr.sizes['frame'] - 1, self.current_frame_idx + 30))

    def set_position(self, position):
        """根据进度条位置设置当前帧"""
        self.update_frame(position)


class ComparisonVideoPlayer(QWidget):
    """一个可以显示处理前后两个视频的播放器"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.before_player = VideoPlayer()
        self.after_player = VideoPlayer()
        self.layout.addWidget(self.before_player)
        self.layout.addWidget(self.after_player)

    def load_video_data(self, before_varr: xr.DataArray, after_varr: xr.DataArray):
        """加载处理前后的视频数据"""
        self.before_player.load_video_data(before_varr)
        self.after_player.load_video_data(after_varr)
        self.before_player.canvas.axes.set_title("处理前")
        self.after_player.canvas.axes.set_title("处理后")


# 重写CNMF的可视化
class CNMFVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.fig, (self.ax_A, self.ax_C) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

    def update_all(self, A, C):
        self.ax_A.clear()
        self.ax_C.clear()
        
        # 假设 A 是 (unit_id, height, width)
        # 假设 C 是 (unit_id, frame)
        
        # 可视化空间部分
        self.ax_A.imshow(A.sum(axis=0), cmap='magma')
        self.ax_A.set_title("CNMF Spatial Footprints")
        self.ax_A.set_xlabel("Width")
        self.ax_A.set_ylabel("Height")

        # 可视化时间部分
        C.T.plot.line(ax=self.ax_C, add_legend=False)
        self.ax_C.set_title("CNMF Temporal Traces")
        self.ax_C.set_xlabel("Frame")
        self.ax_C.set_ylabel("Activity")
        
        self.fig.tight_layout()
        self.canvas.draw()