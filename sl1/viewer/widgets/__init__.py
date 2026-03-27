"""策略查看器UI组件模块。

本模块包含策略查看器的所有PyQt6 UI组件：
- GameTreeWidget: 游戏树控件
- HandRangeMatrixWidget: 手牌矩阵控件
- InfoPanelWidget: 信息面板控件
- StrategyDetailWidget: 策略详情控件
- BoardCardSelector: 公共牌选择器
- ValueHeatmapWidget: 价值热图控件（旧架构，保留兼容性）
- RegretHeatmapWidget: 遗憾值热图控件（Deep CFR架构）
"""

from viewer.widgets.game_tree_widget import GameTreeWidget
from viewer.widgets.hand_matrix_widget import HandRangeMatrixWidget
from viewer.widgets.info_panel_widget import InfoPanelWidget
from viewer.widgets.strategy_detail_widget import StrategyDetailWidget
from viewer.widgets.board_selector_widget import BoardCardSelector
from viewer.widgets.value_heatmap_widget import ValueHeatmapWidget
from viewer.widgets.regret_heatmap_widget import RegretHeatmapWidget

__all__ = [
    'GameTreeWidget',
    'HandRangeMatrixWidget',
    'InfoPanelWidget',
    'StrategyDetailWidget',
    'BoardCardSelector',
    'ValueHeatmapWidget',
    'RegretHeatmapWidget',
]
