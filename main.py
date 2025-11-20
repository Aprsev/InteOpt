import sys
import traceback
from PyQt5.QtWidgets import QApplication

# 导入启动窗口类
try:
    from startup_window import StartupWindow
except ImportError:
    traceback.print_exc()
    print("错误: 无法导入 startup_window.py 中的 StartupWindow 类。请确保该文件存在于项目根目录下。")
    sys.exit(1)

def log_uncaught_exceptions(ex_type, ex_value, ex_traceback):
    """
    全局异常捕获函数，用于在终端打印所有未捕获的 PyQt5 异常。
    """
    print("--------------------------------------------------")
    print("未捕获的全局异常:")
    print(f"类型: {ex_type.__name__}")
    print(f"值: {ex_value}")
    print("--- 追踪栈 ---")
    traceback.print_tb(ex_traceback)
    print("--------------------------------------------------")
    
    # 调用默认的异常处理
    sys.__excepthook__(ex_type, ex_value, ex_traceback)


if __name__ == "__main__":
    # 设置全局异常捕获
    sys.excepthook = log_uncaught_exceptions
    
    # 创建 QApplication 实例
    app = QApplication(sys.argv)
    
    print("程序启动...")
    
    try:
        # 实例化启动窗口
        window = StartupWindow()
        window.show()
        print("启动窗口已显示。")
        
        # 运行应用程序的主事件循环
        sys.exit(app.exec_())
        
    except Exception as e:
        # 捕获启动过程中可能发生的错误
        print(f"程序启动失败，发生致命错误: {e}")
        traceback.print_exc()
        sys.exit(1)