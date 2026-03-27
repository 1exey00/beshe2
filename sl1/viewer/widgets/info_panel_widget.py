"""信息面板控件模块。

本模块实现了游戏状态信息的显示控件，支持：
- 显示公共牌
- 显示当前玩家编号
- 显示总玩家数
- 显示当前底池大小
- 显示当前玩家筹码
- 显示当前游戏阶段

需求引用:
- 需求 5.1: 显示当前公共牌
- 需求 5.2: 显示当前玩家编号
- 需求 5.3: 显示总玩家数
- 需求 5.4: 显示当前底池大小
- 需求 5.5: 显示当前玩家筹码
- 需求 5.6: 显示当前游戏阶段
"""

from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPalette

from models.core import Card, GameStage
from viewer.models import NodeState, GameTreeNode


# 游戏阶段的中文名称
STAGE_NAMES = {
    GameStage.PREFLOP: "翻牌前",
    GameStage.FLOP: "翻牌",
    GameStage.TURN: "转牌",
    GameStage.RIVER: "河牌",
}

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


class InfoPanelWidget(QWidget):
    """信息面板控件 - 显示游戏状态信息。
    
    该控件显示以下信息：
    1. 公共牌（如果有）
    2. 当前玩家编号
    3. 总玩家数
    4. 当前底池大小
    5. 当前玩家筹码
    6. 当前游戏阶段
    
    Attributes:
        _node_state: 当前节点状态
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """初始化信息面板控件。
        
        Args:
            parent: 父控件
        """
        super().__init__(parent)
        
        # 当前节点状态
        self._node_state: Optional[NodeState] = None
        
        # 初始化UI组件
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """设置UI布局。"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 游戏阶段区域
        self._stage_group = self._create_stage_group()
        main_layout.addWidget(self._stage_group)
        
        # 公共牌区域
        self._board_group = self._create_board_group()
        main_layout.addWidget(self._board_group)
        
        # 玩家信息区域
        self._player_group = self._create_player_group()
        main_layout.addWidget(self._player_group)
        
        # 底池和筹码区域
        self._chips_group = self._create_chips_group()
        main_layout.addWidget(self._chips_group)
        
        # 添加弹性空间
        main_layout.addStretch()
        
        # 设置大小策略
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )
    
    def _create_stage_group(self) -> QGroupBox:
        """创建游戏阶段显示区域。"""
        group = QGroupBox("游戏阶段")
        layout = QVBoxLayout(group)
        
        self._stage_label = QLabel("--")
        self._stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 设置大字体
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self._stage_label.setFont(font)
        
        layout.addWidget(self._stage_label)
        
        return group
    
    def _create_board_group(self) -> QGroupBox:
        """创建公共牌显示区域。"""
        group = QGroupBox("公共牌")
        layout = QHBoxLayout(group)
        layout.setSpacing(5)
        
        # 创建5个牌位标签
        self._board_card_labels: List[QLabel] = []
        for i in range(5):
            card_label = self._create_card_label()
            self._board_card_labels.append(card_label)
            layout.addWidget(card_label)
        
        return group
    
    def _create_card_label(self) -> QLabel:
        """创建单个牌位标签。"""
        label = QLabel("--")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(40, 50)
        label.setMaximumSize(50, 65)
        
        # 设置边框样式
        label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
                padding: 5px;
            }
        """)
        
        # 设置字体
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        label.setFont(font)
        
        return label
    
    def _create_player_group(self) -> QGroupBox:
        """创建玩家信息显示区域。"""
        group = QGroupBox("玩家信息")
        layout = QVBoxLayout(group)
        
        # 当前玩家
        current_player_layout = QHBoxLayout()
        current_player_layout.addWidget(QLabel("当前玩家:"))
        self._current_player_label = QLabel("--")
        self._current_player_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        current_player_layout.addWidget(self._current_player_label)
        current_player_layout.addStretch()
        layout.addLayout(current_player_layout)
        
        # 总玩家数
        total_players_layout = QHBoxLayout()
        total_players_layout.addWidget(QLabel("总玩家数:"))
        self._total_players_label = QLabel("--")
        self._total_players_label.setStyleSheet("font-weight: bold;")
        total_players_layout.addWidget(self._total_players_label)
        total_players_layout.addStretch()
        layout.addLayout(total_players_layout)
        
        return group
    
    def _create_chips_group(self) -> QGroupBox:
        """创建底池和筹码显示区域。"""
        group = QGroupBox("底池与筹码")
        layout = QVBoxLayout(group)
        
        # 底池
        pot_layout = QHBoxLayout()
        pot_layout.addWidget(QLabel("底池:"))
        self._pot_label = QLabel("$0")
        self._pot_label.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 14px;")
        pot_layout.addWidget(self._pot_label)
        pot_layout.addStretch()
        layout.addLayout(pot_layout)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # 玩家筹码
        self._stack_labels: List[QLabel] = []
        for i in range(2):
            stack_layout = QHBoxLayout()
            stack_layout.addWidget(QLabel(f"玩家{i + 1}筹码:"))
            stack_label = QLabel("$0")
            stack_label.setStyleSheet("font-weight: bold;")
            self._stack_labels.append(stack_label)
            stack_layout.addWidget(stack_label)
            stack_layout.addStretch()
            layout.addLayout(stack_layout)
        
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
    
    def _get_card_color(self, card: Card) -> QColor:
        """获取牌的显示颜色。
        
        Args:
            card: 牌对象
            
        Returns:
            颜色对象
        """
        return SUIT_COLORS.get(card.suit, QColor(0, 0, 0))
    
    def _update_board_cards(self, board_cards: List[Card]) -> None:
        """更新公共牌显示。
        
        Args:
            board_cards: 公共牌列表
        """
        for i, label in enumerate(self._board_card_labels):
            if i < len(board_cards):
                card = board_cards[i]
                card_text = self._format_card(card)
                label.setText(card_text)
                
                # 设置颜色
                color = self._get_card_color(card)
                label.setStyleSheet(f"""
                    QLabel {{
                        border: 2px solid #999;
                        border-radius: 5px;
                        background-color: white;
                        padding: 5px;
                        color: {color.name()};
                    }}
                """)
            else:
                label.setText("--")
                label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #ccc;
                        border-radius: 5px;
                        background-color: #f9f9f9;
                        padding: 5px;
                        color: #999;
                    }
                """)
    
    # ========================================================================
    # 公共方法
    # ========================================================================
    
    def set_node_state(self, state: NodeState) -> None:
        """设置节点状态并更新显示。
        
        Args:
            state: 节点状态对象
        """
        self._node_state = state
        self._update_display()
    
    def set_from_game_tree_node(self, node: GameTreeNode) -> None:
        """从游戏树节点设置状态。
        
        Args:
            node: 游戏树节点
        """
        state = NodeState.from_game_tree_node(node)
        self.set_node_state(state)
    
    def _update_display(self) -> None:
        """更新所有显示内容。"""
        if self._node_state is None:
            self._clear_display()
            return
        
        state = self._node_state
        
        # 更新游戏阶段
        stage_name = STAGE_NAMES.get(state.stage, str(state.stage.value))
        self._stage_label.setText(stage_name)
        
        # 根据阶段设置颜色
        stage_colors = {
            GameStage.PREFLOP: "#9C27B0",  # 紫色
            GameStage.FLOP: "#2196F3",     # 蓝色
            GameStage.TURN: "#FF9800",     # 橙色
            GameStage.RIVER: "#F44336",    # 红色
        }
        color = stage_colors.get(state.stage, "#333")
        self._stage_label.setStyleSheet(f"color: {color};")
        
        # 更新公共牌
        self._update_board_cards(state.board_cards)
        
        # 更新当前玩家
        self._current_player_label.setText(f"玩家{state.current_player + 1}")
        
        # 更新总玩家数
        self._total_players_label.setText(str(state.total_players))
        
        # 更新底池
        self._pot_label.setText(f"${state.pot}")
        
        # 更新玩家筹码
        for i, stack_label in enumerate(self._stack_labels):
            if i < len(state.stacks):
                stack_label.setText(f"${state.stacks[i]}")
                
                # 高亮当前玩家的筹码
                if i == state.current_player:
                    stack_label.setStyleSheet("font-weight: bold; color: #2196F3;")
                else:
                    stack_label.setStyleSheet("font-weight: bold; color: #333;")
            else:
                stack_label.setText("$0")
    
    def _clear_display(self) -> None:
        """清空所有显示内容。"""
        self._stage_label.setText("--")
        self._stage_label.setStyleSheet("")
        
        for label in self._board_card_labels:
            label.setText("--")
            label.setStyleSheet("""
                QLabel {
                    border: 2px solid #ccc;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                    padding: 5px;
                    color: #999;
                }
            """)
        
        self._current_player_label.setText("--")
        self._total_players_label.setText("--")
        self._pot_label.setText("$0")
        
        for label in self._stack_labels:
            label.setText("$0")
            label.setStyleSheet("font-weight: bold; color: #333;")
    
    def clear(self) -> None:
        """清空显示。"""
        self._node_state = None
        self._clear_display()
    
    def get_node_state(self) -> Optional[NodeState]:
        """获取当前节点状态。
        
        Returns:
            节点状态，如果未设置则返回None
        """
        return self._node_state
    
    def refresh(self) -> None:
        """刷新显示。"""
        self._update_display()
