"""策略详情控件模块。

本模块实现了单个手牌详细策略的显示控件，支持：
- 显示手牌标签和类型
- 显示所有可用行动及其概率百分比
- 分别显示每个花色组合的策略
- 动态生成与动作数量匹配的概率条
- 使用新的颜色方案（蓝色-弃牌，绿色-过牌/跟注，红色系-加注类）

需求引用:
- 需求 4.2: 列出所有可用行动及其对应的概率百分比
- 需求 4.3: 分别显示每个花色组合的策略
- 需求 5.1: 动态生成与动作数量匹配的概率条
- 需求 5.2: 颜色映射器支持任意数量的动作类型
"""

from typing import Optional, Dict, List, Tuple
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QProgressBar,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QBrush

from models.core import Card
from viewer.models import HandStrategy, ActionConfig
from viewer.color_mapper import StrategyColorMapper


# 花色符号映射
SUIT_SYMBOLS = {
    'h': '♥',  # 红心
    'd': '♦',  # 方块
    'c': '♣',  # 梅花
    's': '♠',  # 黑桃
}

# 花色颜色
SUIT_COLORS = {
    'h': QColor(220, 20, 60),   # 红色
    'd': QColor(220, 20, 60),   # 红色
    'c': QColor(0, 0, 0),       # 黑色
    's': QColor(0, 0, 0),       # 黑色
}

# 牌面等级映射
RANK_TO_CHAR = {
    14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
    9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'
}


class StrategyDetailWidget(QWidget):
    """策略详情控件 - 显示单个手牌的详细策略。
    
    该控件显示以下信息：
    1. 手牌标签和类型（对子/同花/非同花）
    2. 所有可用行动及其概率百分比（动态生成）
    3. 每个花色组合的具体策略（如果有）
    
    颜色方案（根据设计文档需求6）：
    - 蓝色系: 弃牌（FOLD）
    - 绿色系: 过牌/跟注（CHECK/CALL）
    - 红色系: 加注类动作（RAISE_SMALL浅红、RAISE_BIG深红、ALL_IN最深红）
    
    Attributes:
        _hand_strategy: 当前手牌策略
        _color_mapper: 颜色映射器
        _action_config: 动作配置（用于动态生成概率条）
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """初始化策略详情控件。
        
        Args:
            parent: 父控件
        """
        super().__init__(parent)
        
        # 当前手牌策略
        self._hand_strategy: Optional[HandStrategy] = None
        
        # 颜色映射器（使用新的颜色方案）
        self._color_mapper = StrategyColorMapper()
        
        # 动作配置（用于动态生成概率条）
        self._action_config: Optional[ActionConfig] = None
        
        # 初始化UI组件
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """设置UI布局。"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 手牌信息区域
        self._hand_info_group = self._create_hand_info_group()
        main_layout.addWidget(self._hand_info_group)
        
        # 策略概率区域
        self._strategy_group = self._create_strategy_group()
        main_layout.addWidget(self._strategy_group)
        
        # 花色组合详情区域
        self._combinations_group = self._create_combinations_group()
        main_layout.addWidget(self._combinations_group)
        
        # 添加弹性空间
        main_layout.addStretch()
        
        # 设置大小策略
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )
    
    def _create_hand_info_group(self) -> QGroupBox:
        """创建手牌信息显示区域。"""
        group = QGroupBox("手牌信息")
        layout = QVBoxLayout(group)
        
        # 手牌标签（大字体显示）
        self._hand_label = QLabel("--")
        self._hand_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self._hand_label.setFont(font)
        layout.addWidget(self._hand_label)
        
        # 手牌类型
        self._hand_type_label = QLabel("--")
        self._hand_type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hand_type_label.setStyleSheet("color: #666;")
        layout.addWidget(self._hand_type_label)
        
        # 组合数量
        self._combinations_count_label = QLabel("--")
        self._combinations_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._combinations_count_label.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(self._combinations_count_label)
        
        return group
    
    def _create_strategy_group(self) -> QGroupBox:
        """创建策略概率显示区域。"""
        group = QGroupBox("策略概率")
        layout = QVBoxLayout(group)
        
        # 策略表格
        self._strategy_table = QTableWidget()
        self._strategy_table.setColumnCount(3)
        self._strategy_table.setHorizontalHeaderLabels(["行动", "概率", ""])
        
        # 设置列宽
        header = self._strategy_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._strategy_table.setColumnWidth(1, 70)
        
        # 设置表格属性
        self._strategy_table.setAlternatingRowColors(True)
        self._strategy_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._strategy_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._strategy_table.verticalHeader().setVisible(False)
        self._strategy_table.setMaximumHeight(200)
        
        layout.addWidget(self._strategy_table)
        
        return group
    
    def _create_combinations_group(self) -> QGroupBox:
        """创建花色组合详情区域。"""
        group = QGroupBox("花色组合详情")
        
        # 使用滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # 组合内容容器
        self._combinations_container = QWidget()
        self._combinations_layout = QVBoxLayout(self._combinations_container)
        self._combinations_layout.setContentsMargins(5, 5, 5, 5)
        self._combinations_layout.setSpacing(5)
        
        scroll_area.setWidget(self._combinations_container)
        
        layout = QVBoxLayout(group)
        layout.addWidget(scroll_area)
        
        return group
    
    def _format_card(self, card: Card) -> str:
        """格式化单张牌的显示文本。
        
        Args:
            card: 牌对象
            
        Returns:
            格式化的字符串（如"A♠"）
        """
        rank_char = RANK_TO_CHAR.get(card.rank, str(card.rank))
        suit_symbol = SUIT_SYMBOLS.get(card.suit, card.suit)
        return f"{rank_char}{suit_symbol}"
    
    def _format_combination(self, card1: Card, card2: Card) -> str:
        """格式化手牌组合的显示文本。
        
        Args:
            card1: 第一张牌
            card2: 第二张牌
            
        Returns:
            格式化的字符串（如"A♠K♠"）
        """
        return f"{self._format_card(card1)}{self._format_card(card2)}"
    
    def _get_hand_type_text(self, hand_label: str) -> str:
        """获取手牌类型的中文描述。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            类型描述
        """
        if len(hand_label) == 2:
            return "对子"
        elif hand_label.endswith('s'):
            return "同花"
        else:
            return "非同花"
    
    def _get_expected_combinations_count(self, hand_label: str) -> int:
        """获取手牌标签的预期组合数量。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            组合数量
        """
        if len(hand_label) == 2:
            return 6   # 对子: C(4,2) = 6
        elif hand_label.endswith('s'):
            return 4   # 同花: 4种花色
        else:
            return 12  # 非同花: 4×3 = 12
    
    def _get_action_color(self, action: str) -> QColor:
        """获取行动的显示颜色。
        
        使用StrategyColorMapper的新颜色方案：
        - FOLD: 蓝色系 (R < 100, G > 100, B > 200)
        - CHECK/CALL: 绿色系 (R < 100, G > 150, B < 150)
        - RAISE类: 红色系 (R > 180, G < 150, B < 150)
        
        Args:
            action: 行动名称
            
        Returns:
            颜色对象
        """
        # 使用颜色映射器获取颜色
        color = self._color_mapper.get_action_color(action)
        return QColor(color.r, color.g, color.b, color.a)
    
    def _update_strategy_table(self, action_probs: Dict[str, float]) -> None:
        """更新策略概率表格。
        
        动态生成与动作数量匹配的概率条，使用新的颜色方案。
        
        如果设置了动作配置，则按配置中的动作顺序显示；
        否则按概率降序排序显示。
        
        Args:
            action_probs: 行动概率字典
            
        需求引用:
        - 需求 5.1: 动态生成与动作数量匹配的概率条
        - 需求 5.2: 颜色映射器支持任意数量的动作类型
        """
        # 确定动作顺序
        if self._action_config is not None:
            # 使用动作配置中的顺序，确保所有动作都显示
            action_order = self._action_config.action_names
            # 构建完整的动作概率字典（包含配置中的所有动作）
            full_action_probs = {}
            for action in action_order:
                full_action_probs[action] = action_probs.get(action, 0.0)
            # 按概率降序排序，但保持配置中的动作
            sorted_actions = sorted(
                full_action_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            # 没有动作配置时，按概率降序排序
            sorted_actions = sorted(
                action_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
        
        # 设置行数
        self._strategy_table.setRowCount(len(sorted_actions))
        
        for row, (action, prob) in enumerate(sorted_actions):
            # 获取显示名称（如果有动作配置）
            display_name = action
            if self._action_config is not None:
                display_name = self._action_config.get_display_name(action)
            
            # 行动名称
            action_item = QTableWidgetItem(display_name)
            action_color = self._get_action_color(action)
            action_item.setForeground(QBrush(action_color))
            font = action_item.font()
            font.setBold(True)
            action_item.setFont(font)
            self._strategy_table.setItem(row, 0, action_item)
            
            # 概率百分比
            prob_text = f"{prob * 100:.1f}%"
            prob_item = QTableWidgetItem(prob_text)
            prob_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._strategy_table.setItem(row, 1, prob_item)
            
            # 进度条（使用新的颜色方案）
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(int(prob * 100))
            progress_bar.setTextVisible(False)
            progress_bar.setMaximumHeight(15)
            
            # 设置进度条颜色（使用StrategyColorMapper的颜色方案）
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    background-color: #f5f5f5;
                }}
                QProgressBar::chunk {{
                    background-color: {action_color.name()};
                    border-radius: 2px;
                }}
            """)
            
            self._strategy_table.setCellWidget(row, 2, progress_bar)
        
        # 调整行高
        self._strategy_table.resizeRowsToContents()
    
    def _update_combinations_display(
        self, 
        combinations: List[Tuple[Card, Card]],
        combination_strategies: Optional[Dict[str, Dict[str, float]]]
    ) -> None:
        """更新花色组合详情显示。
        
        Args:
            combinations: 花色组合列表
            combination_strategies: 每个组合的策略（可选）
        """
        # 清空现有内容
        while self._combinations_layout.count():
            item = self._combinations_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not combinations:
            no_data_label = QLabel("无可用组合")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_data_label.setStyleSheet("color: #999;")
            self._combinations_layout.addWidget(no_data_label)
            return
        
        # 为每个组合创建显示项
        for card1, card2 in combinations:
            combo_widget = self._create_combination_item(
                card1, card2, combination_strategies
            )
            self._combinations_layout.addWidget(combo_widget)
        
        # 添加弹性空间
        self._combinations_layout.addStretch()
    
    def _create_combination_item(
        self,
        card1: Card,
        card2: Card,
        combination_strategies: Optional[Dict[str, Dict[str, float]]]
    ) -> QWidget:
        """创建单个组合的显示项。
        
        Args:
            card1: 第一张牌
            card2: 第二张牌
            combination_strategies: 组合策略字典
            
        Returns:
            组合显示控件
        """
        widget = QFrame()
        widget.setFrameShape(QFrame.Shape.StyledPanel)
        widget.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 组合标签
        combo_text = self._format_combination(card1, card2)
        combo_label = QLabel(combo_text)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        combo_label.setFont(font)
        
        # 设置花色颜色
        color1 = SUIT_COLORS.get(card1.suit, QColor(0, 0, 0))
        combo_label.setStyleSheet(f"color: {color1.name()};")
        
        layout.addWidget(combo_label)
        
        # 如果有组合特定策略，显示主要行动
        combo_key = f"{self._format_card(card1)}{self._format_card(card2)}"
        if combination_strategies and combo_key in combination_strategies:
            strategy = combination_strategies[combo_key]
            if strategy:
                # 找到概率最高的行动
                dominant_action = max(strategy.items(), key=lambda x: x[1])
                action_name, prob = dominant_action
                
                action_label = QLabel(f"{action_name}: {prob*100:.1f}%")
                action_color = self._get_action_color(action_name)
                action_label.setStyleSheet(f"color: {action_color.name()};")
                layout.addWidget(action_label)
        
        layout.addStretch()
        
        return widget
    
    # ========================================================================
    # 公共方法
    # ========================================================================
    
    def set_hand_strategy(self, strategy: HandStrategy) -> None:
        """设置手牌策略并更新显示。
        
        Args:
            strategy: 手牌策略对象
        """
        self._hand_strategy = strategy
        self._update_display()
    
    def _update_display(self) -> None:
        """更新所有显示内容。"""
        if self._hand_strategy is None:
            self._clear_display()
            return
        
        strategy = self._hand_strategy
        
        # 更新手牌标签
        self._hand_label.setText(strategy.hand_label)
        
        # 更新手牌类型
        hand_type = self._get_hand_type_text(strategy.hand_label)
        self._hand_type_label.setText(hand_type)
        
        # 更新组合数量
        actual_count = len(strategy.combinations)
        expected_count = self._get_expected_combinations_count(strategy.hand_label)
        if actual_count < expected_count:
            count_text = f"{actual_count}/{expected_count} 个组合（部分与公共牌冲突）"
        else:
            count_text = f"{actual_count} 个组合"
        self._combinations_count_label.setText(count_text)
        
        # 更新策略表格
        self._update_strategy_table(strategy.action_probabilities)
        
        # 更新花色组合详情
        self._update_combinations_display(
            strategy.combinations,
            strategy.combination_strategies
        )
    
    def _clear_display(self) -> None:
        """清空所有显示内容。"""
        self._hand_label.setText("--")
        self._hand_type_label.setText("--")
        self._combinations_count_label.setText("--")
        
        self._strategy_table.setRowCount(0)
        
        # 清空组合显示
        while self._combinations_layout.count():
            item = self._combinations_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        no_data_label = QLabel("请选择一个手牌")
        no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_data_label.setStyleSheet("color: #999;")
        self._combinations_layout.addWidget(no_data_label)
    
    def clear(self) -> None:
        """清空显示。"""
        self._hand_strategy = None
        self._clear_display()
    
    def get_hand_strategy(self) -> Optional[HandStrategy]:
        """获取当前手牌策略。
        
        Returns:
            手牌策略，如果未设置则返回None
        """
        return self._hand_strategy
    
    def set_color_mapper(self, mapper: StrategyColorMapper) -> None:
        """设置颜色映射器。
        
        Args:
            mapper: 颜色映射器
        """
        self._color_mapper = mapper
        self._update_display()
    
    def set_action_config(self, config: Optional[ActionConfig]) -> None:
        """设置动作配置。
        
        动作配置用于动态生成与模型动作数量匹配的概率条。
        
        Args:
            config: 动作配置对象，如果为None则使用策略中的动作
            
        需求引用:
        - 需求 5.1: 动态生成与动作数量匹配的概率条
        """
        self._action_config = config
        self._update_display()
    
    def get_action_config(self) -> Optional[ActionConfig]:
        """获取当前动作配置。
        
        Returns:
            动作配置对象，如果未设置则返回None
        """
        return self._action_config
    
    def refresh(self) -> None:
        """刷新显示。"""
        self._update_display()
