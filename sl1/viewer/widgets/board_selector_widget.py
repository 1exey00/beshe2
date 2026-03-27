"""公共牌选择器控件模块。

本模块实现了公共牌选择界面，支持：
- 选择翻牌、转牌、河牌
- 发出 board_changed 信号
- 显示已选择的公共牌

需求引用:
- 需求 6.1: 切换到"BoardCards"标签页时显示公共牌设置界面
- 需求 6.2: 选择特定的公共牌组合时更新手牌矩阵
- 需求 6.3: 设置的公共牌与某些手牌冲突时禁用或标记这些手牌组合
"""

from typing import Optional, List, Set
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QGroupBox,
    QPushButton,
    QFrame,
    QSizePolicy,
    QScrollArea,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor

from models.core import Card


# 牌面等级（从高到低）
RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

# 牌面等级到数值的映射
RANK_TO_VALUE = {
    'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
    '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
}

# 数值到牌面等级的映射
VALUE_TO_RANK = {v: k for k, v in RANK_TO_VALUE.items()}

# 花色
SUITS = ['s', 'h', 'd', 'c']  # 黑桃、红心、方块、梅花

# 花色符号
SUIT_SYMBOLS = {
    's': '♠',  # 黑桃
    'h': '♥',  # 红心
    'd': '♦',  # 方块
    'c': '♣',  # 梅花
}

# 花色颜色
SUIT_COLORS = {
    's': '#000000',  # 黑色
    'h': '#DC143C',  # 红色
    'd': '#DC143C',  # 红色
    'c': '#000000',  # 黑色
}

# 花色中文名称
SUIT_NAMES = {
    's': '黑桃',
    'h': '红心',
    'd': '方块',
    'c': '梅花',
}


class CardButton(QPushButton):
    """单张牌按钮控件。
    
    用于在公共牌选择器中显示和选择单张牌。
    
    Attributes:
        card: 对应的Card对象
        _is_selected: 是否被选中
    """
    
    def __init__(
        self, 
        rank: str, 
        suit: str, 
        parent: Optional[QWidget] = None
    ):
        """初始化牌按钮。
        
        Args:
            rank: 牌面等级（A, K, Q, J, T, 9-2）
            suit: 花色（s, h, d, c）
            parent: 父控件
        """
        super().__init__(parent)
        
        self._rank = rank
        self._suit = suit
        self._is_selected = False
        
        # 创建Card对象
        self.card = Card(rank=RANK_TO_VALUE[rank], suit=suit)
        
        # 设置按钮文本
        text = f"{rank}{SUIT_SYMBOLS[suit]}"
        self.setText(text)
        
        # 设置固定大小
        self.setFixedSize(45, 55)
        
        # 设置字体
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.setFont(font)
        
        # 更新样式
        self._update_style()
    
    def _update_style(self) -> None:
        """更新按钮样式。"""
        color = SUIT_COLORS[self._suit]
        
        if self._is_selected:
            # 选中状态
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: #E3F2FD;
                    border: 3px solid #2196F3;
                    border-radius: 5px;
                    color: {color};
                }}
                QPushButton:hover {{
                    background-color: #BBDEFB;
                }}
            """)
        else:
            # 未选中状态
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: white;
                    border: 2px solid #ccc;
                    border-radius: 5px;
                    color: {color};
                }}
                QPushButton:hover {{
                    background-color: #f5f5f5;
                    border-color: #999;
                }}
                QPushButton:disabled {{
                    background-color: #e0e0e0;
                    border-color: #bbb;
                    color: #999;
                }}
            """)
    
    def set_selected(self, selected: bool) -> None:
        """设置选中状态。
        
        Args:
            selected: 是否选中
        """
        self._is_selected = selected
        self._update_style()
    
    def is_selected(self) -> bool:
        """获取选中状态。
        
        Returns:
            是否选中
        """
        return self._is_selected
    
    def get_rank(self) -> str:
        """获取牌面等级。"""
        return self._rank
    
    def get_suit(self) -> str:
        """获取花色。"""
        return self._suit


class SelectedCardLabel(QLabel):
    """已选择的牌显示标签。
    
    用于在公共牌选择器中显示已选择的牌。
    """
    
    def __init__(
        self, 
        index: int,
        parent: Optional[QWidget] = None
    ):
        """初始化已选择牌标签。
        
        Args:
            index: 牌位索引（0-4）
            parent: 父控件
        """
        super().__init__(parent)
        
        self._index = index
        self._card: Optional[Card] = None
        
        # 设置固定大小
        self.setFixedSize(50, 65)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 设置字体
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.setFont(font)
        
        # 初始化为空状态
        self._update_display()
    
    def _update_display(self) -> None:
        """更新显示。"""
        if self._card is None:
            self.setText("--")
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #ccc;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                    color: #999;
                }
            """)
        else:
            rank_char = VALUE_TO_RANK.get(self._card.rank, str(self._card.rank))
            suit_symbol = SUIT_SYMBOLS.get(self._card.suit, self._card.suit)
            self.setText(f"{rank_char}{suit_symbol}")
            
            color = SUIT_COLORS.get(self._card.suit, '#000')
            self.setStyleSheet(f"""
                QLabel {{
                    border: 2px solid #999;
                    border-radius: 5px;
                    background-color: white;
                    color: {color};
                }}
            """)
    
    def set_card(self, card: Optional[Card]) -> None:
        """设置牌。
        
        Args:
            card: Card对象，None表示清空
        """
        self._card = card
        self._update_display()
    
    def get_card(self) -> Optional[Card]:
        """获取牌。
        
        Returns:
            Card对象，如果为空则返回None
        """
        return self._card
    
    def clear(self) -> None:
        """清空牌。"""
        self._card = None
        self._update_display()



class BoardCardSelector(QWidget):
    """公共牌选择器控件。
    
    该控件提供以下功能：
    1. 显示52张牌的选择面板
    2. 支持选择最多5张公共牌（翻牌3张 + 转牌1张 + 河牌1张）
    3. 显示已选择的公共牌
    4. 发出 board_changed 信号
    
    Signals:
        board_changed(list): 公共牌改变时发出，参数为Card列表
    
    Attributes:
        _selected_cards: 已选择的牌列表
        _card_buttons: 所有牌按钮的字典
        _selected_labels: 已选择牌的显示标签列表
    """
    
    # 定义信号
    board_changed = pyqtSignal(list)  # 发送Card列表
    
    # 最大公共牌数量
    MAX_BOARD_CARDS = 5
    
    def __init__(self, parent: Optional[QWidget] = None):
        """初始化公共牌选择器。
        
        Args:
            parent: 父控件
        """
        super().__init__(parent)
        
        # 已选择的牌列表
        self._selected_cards: List[Card] = []
        
        # 牌按钮字典 {(rank, suit): CardButton}
        self._card_buttons: dict = {}
        
        # 已选择牌的显示标签
        self._selected_labels: List[SelectedCardLabel] = []
        
        # 初始化UI
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """设置UI布局。"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # 已选择的公共牌显示区域
        selected_group = self._create_selected_group()
        main_layout.addWidget(selected_group)
        
        # 牌选择面板
        selection_group = self._create_selection_group()
        main_layout.addWidget(selection_group)
        
        # 操作按钮区域
        buttons_layout = self._create_buttons_layout()
        main_layout.addLayout(buttons_layout)
        
        # 添加弹性空间
        main_layout.addStretch()
        
        # 设置大小策略
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )
    
    def _create_selected_group(self) -> QGroupBox:
        """创建已选择公共牌显示区域。"""
        group = QGroupBox("已选择的公共牌")
        layout = QHBoxLayout(group)
        layout.setSpacing(10)
        
        # 翻牌标签
        flop_label = QLabel("翻牌:")
        flop_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(flop_label)
        
        # 翻牌3张
        for i in range(3):
            label = SelectedCardLabel(i, self)
            self._selected_labels.append(label)
            layout.addWidget(label)
        
        # 分隔符
        layout.addSpacing(10)
        
        # 转牌标签
        turn_label = QLabel("转牌:")
        turn_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(turn_label)
        
        # 转牌1张
        label = SelectedCardLabel(3, self)
        self._selected_labels.append(label)
        layout.addWidget(label)
        
        # 分隔符
        layout.addSpacing(10)
        
        # 河牌标签
        river_label = QLabel("河牌:")
        river_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(river_label)
        
        # 河牌1张
        label = SelectedCardLabel(4, self)
        self._selected_labels.append(label)
        layout.addWidget(label)
        
        layout.addStretch()
        
        return group
    
    def _create_selection_group(self) -> QGroupBox:
        """创建牌选择面板。"""
        group = QGroupBox("选择公共牌")
        layout = QVBoxLayout(group)
        
        # 按花色分组显示
        for suit in SUITS:
            suit_layout = QHBoxLayout()
            suit_layout.setSpacing(3)
            
            # 花色标签
            suit_label = QLabel(f"{SUIT_NAMES[suit]}:")
            suit_label.setMinimumWidth(50)
            suit_label.setStyleSheet(f"font-weight: bold; color: {SUIT_COLORS[suit]};")
            suit_layout.addWidget(suit_label)
            
            # 该花色的所有牌
            for rank in RANKS:
                btn = CardButton(rank, suit, self)
                btn.clicked.connect(
                    lambda checked, r=rank, s=suit: self._on_card_clicked(r, s)
                )
                self._card_buttons[(rank, suit)] = btn
                suit_layout.addWidget(btn)
            
            suit_layout.addStretch()
            layout.addLayout(suit_layout)
        
        return group
    
    def _create_buttons_layout(self) -> QHBoxLayout:
        """创建操作按钮区域。"""
        layout = QHBoxLayout()
        
        # 清空按钮
        clear_btn = QPushButton("清空全部")
        clear_btn.setMinimumWidth(100)
        clear_btn.clicked.connect(self.clear_all)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        layout.addWidget(clear_btn)
        
        # 清空最后一张按钮
        clear_last_btn = QPushButton("撤销最后一张")
        clear_last_btn.setMinimumWidth(120)
        clear_last_btn.clicked.connect(self.clear_last)
        clear_last_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        layout.addWidget(clear_last_btn)
        
        layout.addStretch()
        
        # 显示当前选择数量
        self._count_label = QLabel("已选择: 0/5")
        self._count_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self._count_label)
        
        return layout
    
    def _on_card_clicked(self, rank: str, suit: str) -> None:
        """处理牌按钮点击事件。
        
        Args:
            rank: 牌面等级
            suit: 花色
        """
        btn = self._card_buttons.get((rank, suit))
        if btn is None:
            return
        
        card = btn.card
        
        if btn.is_selected():
            # 取消选择
            self._remove_card(card)
        else:
            # 选择牌
            if len(self._selected_cards) < self.MAX_BOARD_CARDS:
                self._add_card(card)
    
    def _add_card(self, card: Card) -> None:
        """添加一张牌到已选择列表。
        
        Args:
            card: 要添加的牌
        """
        if card in self._selected_cards:
            return
        
        if len(self._selected_cards) >= self.MAX_BOARD_CARDS:
            return
        
        self._selected_cards.append(card)
        
        # 更新按钮状态
        rank = VALUE_TO_RANK.get(card.rank, str(card.rank))
        btn = self._card_buttons.get((rank, card.suit))
        if btn:
            btn.set_selected(True)
        
        # 更新显示
        self._update_display()
        
        # 发出信号
        self.board_changed.emit(self._selected_cards.copy())
    
    def _remove_card(self, card: Card) -> None:
        """从已选择列表中移除一张牌。
        
        Args:
            card: 要移除的牌
        """
        if card not in self._selected_cards:
            return
        
        self._selected_cards.remove(card)
        
        # 更新按钮状态
        rank = VALUE_TO_RANK.get(card.rank, str(card.rank))
        btn = self._card_buttons.get((rank, card.suit))
        if btn:
            btn.set_selected(False)
        
        # 更新显示
        self._update_display()
        
        # 发出信号
        self.board_changed.emit(self._selected_cards.copy())
    
    def _update_display(self) -> None:
        """更新显示。"""
        # 更新已选择牌的显示
        for i, label in enumerate(self._selected_labels):
            if i < len(self._selected_cards):
                label.set_card(self._selected_cards[i])
            else:
                label.clear()
        
        # 更新计数标签
        count = len(self._selected_cards)
        self._count_label.setText(f"已选择: {count}/{self.MAX_BOARD_CARDS}")
        
        # 根据数量设置颜色
        if count == 0:
            self._count_label.setStyleSheet("font-weight: bold; color: #666;")
        elif count == 3:
            self._count_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        elif count == 4:
            self._count_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        elif count == 5:
            self._count_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        else:
            self._count_label.setStyleSheet("font-weight: bold; color: #666;")
    
    # ========================================================================
    # 公共方法
    # ========================================================================
    
    def get_selected_cards(self) -> List[Card]:
        """获取已选择的公共牌列表。
        
        Returns:
            Card对象列表
        """
        return self._selected_cards.copy()
    
    def set_selected_cards(self, cards: List[Card]) -> None:
        """设置已选择的公共牌。
        
        Args:
            cards: Card对象列表
        """
        # 清空当前选择
        self._clear_selection_state()
        
        # 添加新的选择
        for card in cards[:self.MAX_BOARD_CARDS]:
            if card not in self._selected_cards:
                self._selected_cards.append(card)
                
                # 更新按钮状态
                rank = VALUE_TO_RANK.get(card.rank, str(card.rank))
                btn = self._card_buttons.get((rank, card.suit))
                if btn:
                    btn.set_selected(True)
        
        # 更新显示
        self._update_display()
        
        # 发出信号
        self.board_changed.emit(self._selected_cards.copy())
    
    def _clear_selection_state(self) -> None:
        """清空选择状态（不发出信号）。"""
        # 重置所有按钮状态
        for btn in self._card_buttons.values():
            btn.set_selected(False)
        
        # 清空已选择列表
        self._selected_cards.clear()
    
    def clear_all(self) -> None:
        """清空所有已选择的牌。"""
        self._clear_selection_state()
        self._update_display()
        self.board_changed.emit([])
    
    def clear_last(self) -> None:
        """清空最后一张已选择的牌。"""
        if self._selected_cards:
            card = self._selected_cards[-1]
            self._remove_card(card)
    
    def get_conflicting_cards(self) -> Set[Card]:
        """获取与已选择公共牌冲突的牌集合。
        
        Returns:
            冲突的Card对象集合
        """
        return set(self._selected_cards)
    
    def is_card_selected(self, card: Card) -> bool:
        """检查指定的牌是否已被选择。
        
        Args:
            card: 要检查的牌
            
        Returns:
            是否已被选择
        """
        return card in self._selected_cards
    
    def get_board_count(self) -> int:
        """获取已选择的公共牌数量。
        
        Returns:
            公共牌数量
        """
        return len(self._selected_cards)
    
    def can_add_more(self) -> bool:
        """检查是否还能添加更多公共牌。
        
        Returns:
            是否还能添加
        """
        return len(self._selected_cards) < self.MAX_BOARD_CARDS
    
    def refresh(self) -> None:
        """刷新显示。"""
        self._update_display()
