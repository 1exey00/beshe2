"""策略查看器模块 - 用于可视化和分析德州扑克AI策略。

本模块提供以下功能：
- 游戏树导航和浏览
- 13x13手牌范围矩阵可视化
- 策略概率分布分析
- 策略数据导出

使用方法:
    # 方式1: 使用命令行脚本
    python run_viewer.py
    python run_viewer.py --checkpoint path/to/checkpoint.pt
    
    # 方式2: 作为模块运行
    python -m viewer
    python -m viewer --checkpoint path/to/checkpoint.pt
    
    # 方式3: 在代码中调用
    from viewer import run_viewer
    run_viewer()  # 或 run_viewer(checkpoint_path="path/to/checkpoint.pt")

需求引用:
- 需求 1.1: 加载训练好的模型检查点
"""

__version__ = "1.0.0"

# 导出核心组件
from viewer.models import GameTreeNode, HandStrategy, NodeState, NodeType
from viewer.model_loader import ModelLoader, ModelMetadata
from viewer.game_tree import GameTreeNavigator
from viewer.hand_range import HandRangeCalculator, HAND_LABELS_MATRIX
from viewer.strategy_calculator import StrategyCalculator, StrategyResult
from viewer.color_mapper import StrategyColorMapper, Color
from viewer.controller import StrategyViewerController, ViewerState

# UI组件（需要PyQt6）
try:
    from viewer.widgets import (
        GameTreeWidget,
        HandRangeMatrixWidget,
        InfoPanelWidget,
        StrategyDetailWidget,
        BoardCardSelector,
    )
    from viewer.main_window import MainWindow, create_main_window
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False
    GameTreeWidget = None
    HandRangeMatrixWidget = None
    InfoPanelWidget = None
    StrategyDetailWidget = None
    BoardCardSelector = None
    MainWindow = None
    create_main_window = None

__all__ = [
    # 数据模型
    "GameTreeNode",
    "HandStrategy", 
    "NodeState",
    "NodeType",
    # 模型加载
    "ModelLoader",
    "ModelMetadata",
    # 游戏树
    "GameTreeNavigator",
    # 手牌范围
    "HandRangeCalculator",
    "HAND_LABELS_MATRIX",
    # 策略计算
    "StrategyCalculator",
    "StrategyResult",
    # 颜色映射
    "StrategyColorMapper",
    "Color",
    # 主控制器
    "StrategyViewerController",
    "ViewerState",
    # UI组件
    "GameTreeWidget",
    "HandRangeMatrixWidget",
    "InfoPanelWidget",
    "StrategyDetailWidget",
    "BoardCardSelector",
    # 主窗口
    "MainWindow",
    "create_main_window",
    # 启动函数
    "run_viewer",
]


def run_viewer(checkpoint_path: str = None) -> int:
    """启动策略查看器GUI应用程序。
    
    这是一个便捷函数，用于在代码中启动策略查看器。
    
    Args:
        checkpoint_path: 可选的检查点文件路径，如果提供则自动加载
        
    Returns:
        应用程序退出码（0表示正常退出）
        
    Raises:
        ImportError: 如果PyQt6未安装
        FileNotFoundError: 如果指定的检查点文件不存在
        
    Example:
        >>> from viewer import run_viewer
        >>> run_viewer()  # 启动空白查看器
        >>> run_viewer("checkpoints/model.pt")  # 启动并加载模型
    """
    import sys
    from pathlib import Path
    
    # 检查PyQt6
    if not _HAS_PYQT6:
        raise ImportError(
            "PyQt6未安装。请运行 'pip install PyQt6' 安装。"
        )
    
    # 验证检查点文件
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        if not path.is_file():
            raise ValueError(f"指定的路径不是文件: {checkpoint_path}")
        checkpoint_path = str(path.resolve())
    
    from PyQt6.QtWidgets import QApplication
    
    # 创建应用程序（如果尚未创建）
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setApplicationName("策略查看器")
        app.setApplicationVersion(__version__)
        app.setOrganizationName("Texas Hold'em AI")
        app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = create_main_window(checkpoint_path=checkpoint_path)
    window.show()
    
    # 运行事件循环
    return app.exec()
