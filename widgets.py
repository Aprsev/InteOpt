from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QPushButton, QGroupBox
)
from typing import Dict, Any

from .config import STEP_NAMES

class ParamPanel(QGroupBox):
    """一个可动态生成参数设置控件的面板"""

    def __init__(self, step_name: str, params: Dict[str, Any], parent=None):
        super().__init__(f"参数设置 - {step_name}", parent)
        self.params = params
        self.inputs = {}
        self.layout = QFormLayout()

        for key, value in params.items():
            if isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(0, 1000)
                widget.setValue(value)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(0.0, 1000.0)
                widget.setValue(value)
            elif isinstance(value, str):
                widget = QLineEdit(value)
            else:
                continue

            self.layout.addRow(QLabel(key), widget)
            self.inputs[key] = widget

        self.setLayout(self.layout)

    def get_params(self) -> Dict[str, Any]:
        """获取当前面板上所有控件的值"""
        updated_params = self.params.copy()
        for key, widget in self.inputs.items():
            if isinstance(widget, QSpinBox):
                updated_params[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                updated_params[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                updated_params[key] = widget.text()
        return updated_params


class StepNavigation(QWidget):
    """步骤导航按钮"""

    def __init__(self, step_ids: list, parent=None):
        super().__init__(parent)
        self.step_ids = step_ids
        self.layout = QVBoxLayout(self)

        # 创建下拉列表
        self.nav_combo = QComboBox()
        self.nav_combo.addItems([STEP_NAMES[sid] for sid in step_ids])
        self.layout.addWidget(self.nav_combo)

        # 创建按钮
        self.next_button = QPushButton("下一步")
        self.prev_button = QPushButton("上一步")
        self.run_button = QPushButton("运行当前步骤")
        self.layout.addWidget(self.next_button)
        self.layout.addWidget(self.prev_button)
        self.layout.addWidget(self.run_button)

    def set_main_window(self, main_window):
        self.nav_combo.currentIndexChanged.connect(main_window.jump_to_step)
        self.next_button.clicked.connect(main_window.next_step)
        self.prev_button.clicked.connect(main_window.prev_step)
        self.run_button.clicked.connect(main_window.run_current_step)