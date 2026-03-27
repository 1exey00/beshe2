"""遗憾值热图控件模块。

本模块实现了13x13手牌遗憾值热图的显示：
- 使用颜色编码表示遗憾值高低（蓝色=低遗憾，红色=高遗憾）
- 支持鼠标悬停显示具体遗憾值
- 支持公共牌过滤

需求引用:
- 需求 9.2: 显示13x13手牌范围矩阵的遗憾值估计热图
- 需求 9.3: 使用颜色编码表示遗憾值高低
- 需求 9.4: 显示该手牌的具体遗憾值估计数值
"""

from typing import Dict, Optional, List
from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QPainter, QFont, QMouseEvent, QPaintEvent

from viewer.regret_calculator import HandRegret, RegretResult


# 手牌标签矩阵（13x13）
RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

HAND_LABELS = [
    ['AA',  'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s'],
    ['AKo', 'KK',  'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s'],
    ['AQo', 'KQo', 'QQ',  'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s'],
    ['AJo', 'KJo', 'QJo', 'JJ',  'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s'],
    ['ATo', 'KTo', 'QTo', 'JTo', 'TT',  'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s'],
    ['A9o', 'K9o', 'Q9o', 'J9o', 'T9o', '99',  '98s', '97s', '96s', '95s', '94s', '93s', '92s'],
    ['A8o', 'K8o', 'Q8o', 'J8o', 'T8o', '98o', '88',  '87s', '86s', '85s', '84s', '83s', '82s'],
    ['A7o', 'K7o', 'Q7o', 'J7o', 'T7o', '97o', '87o', '77',  '76s', '75s', '74s', '73s', '72s'],
    ['A6o', 'K6o', 'Q6o', 'J6o', 'T6o', '96o', '86o', '76o', '66',  '65s', '64s', '63s', '62s'],
    ['A5o', 'K5o', 'Q5o', 'J5o', 'T5o', '95o', '85o', '75o', '65o', '55',  '54s', '53s', '52s'],
    ['A4o', 'K4o', 'Q4o', 'J4o', 'T4o', '94o', '84o', '74o', '64o', '54o', '44',  '43s', '42s'],
    ['A3o', 'K3o', 'Q3o', 'J3o', 'T3o', '93o', '83o', '73o', '63o', '53o', '43o', '33',  '32s'],
    ['A2o', 'K2o', 'Q2o', 'J2o', 'T2o', '92o', '82o', '72o', '62o', '52o', '42o', '32o', '22'],
]


class RegretHeatmapCell(QWidget):
    """遗憾值热图单元格控件。"""
    
    clicked = pyqtSignal(str)  # 点击时发出手牌标签
    hovered = pyqtSignal(str)  # 悬停时发出手牌标签
    
    def __init__(self, hand_label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._hand_label = hand_label
        self._regret = 0.0
        self._regrets_dict: Dict[str, float] = {}
        self._is_blocked = False
        self._color = QColor(200, 200, 200)
        self._is_hovered = False
        
        self.setMinimumSize(40, 40)
        self.setMouseTracking(True)
    
    def set_regret(self, regret: float, regrets_dict: Dict[str, float],
                   min_regret: float, max_regret: float) -> None:
        """设置遗憾值并更新颜色。
        
        Args:
            regret: 总遗憾值（正遗憾值的和）
            regrets_dict: 各动作的遗憾值字典
            min_regret: 最小遗憾值（用于归一化）
            max_regret: 最大遗憾值（用于归一化）
        """
        self._regret = regret
        self._regrets_dict = regrets_dict
        self._is_blocked = False
        
        # 归一化遗憾值到 [0, 1]
        if max_regret > min_regret:
            normalized = (regret - min_regret) / (max_regret - min_regret)
        else:
            normalized = 0.5
        
        # 颜色映射：蓝色(低遗憾) -> 白色(中) -> 红色(高遗憾)
        self._color = self._regret_to_color(normalized)
        self.update()
    
    def set_blocked(self) -> None:
        """设置为被阻挡状态。"""
        self._is_blocked = True
        self._regret = 0.0
        self._regrets_dict = {}
        self._color = QColor(100, 100, 100)
        self.update()
    
    def set_unavailable(self) -> None:
        """设置为不可用状态（没有遗憾网络）。"""
        self._is_blocked = False
        self._regret = 0.0
        self._regrets_dict = {}
        self._color = QColor(180, 180, 180)
        self.update()
    
    def _regret_to_color(self, normalized: float) -> QColor:
        """将归一化遗憾值转换为颜色。
        
        Args:
            normalized: 归一化遗憾值 [0, 1]
            
        Returns:
            颜色
        """
        # 蓝色 -> 白色 -> 红色
        if normalized < 0.5:
            # 蓝色到白色
            t = normalized * 2
            r = int(50 + 205 * t)
            g = int(100 + 155 * t)
            b = 255
        else:
            # 白色到红色
            t = (normalized - 0.5) * 2
            r = 255
            g = int(255 - 155 * t)
            b = int(255 - 205 * t)
        
        return QColor(r, g, b)
    
    @property
    def hand_label(self) -> str:
        return self._hand_label
    
    @property
    def regret(self) -> float:
        return self._regret
    
    @property
    def regrets_dict(self) -> Dict[str, float]:
        return self._regrets_dict
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """绘制单元格。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制背景
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(rect, self._color)
        
        # 绘制边框
        if self._is_hovered:
            painter.setPen(QColor(0, 0, 0))
            painter.drawRect(rect)
        
        # 绘制手牌标签
        painter.setPen(QColor(0, 0, 0) if not self._is_blocked else QColor(150, 150, 150))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._hand_label)
    
    def enterEvent(self, event) -> None:
        """鼠标进入事件。"""
        self._is_hovered = True
        self.update()
        self.hovered.emit(self._hand_label)
    
    def leaveEvent(self, event) -> None:
        """鼠标离开事件。"""
        self._is_hovered = False
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """鼠标点击事件。"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._hand_label)


class RegretHeatmapWidget(QWidget):
    """遗憾值热图控件 - 显示13x13手牌遗憾值热图。
    
    Signals:
        cell_hovered(str): 鼠标悬停在单元格上时发出手牌标签
        cell_clicked(str): 单元格被点击时发出手牌标签
    """
    
    cell_hovered = pyqtSignal(str)
    cell_clicked = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._cells: Dict[str, RegretHeatmapCell] = {}
        self._regret_result: Optional[RegretResult] = None
        self._is_available = False
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """设置UI布局。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 标题
        title_label = QLabel("手牌遗憾值热图（Regret Network）")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #f5f5f5;
                border-bottom: 1px solid #ddd;
            }
        """)
        layout.addWidget(title_label)
        
        # 状态标签
        self._status_label = QLabel("遗憾网络不可用")
        self._status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                padding: 5px;
            }
        """)
        layout.addWidget(self._status_label)
        
        # 矩阵容器
        matrix_widget = QWidget()
        matrix_layout = QGridLayout(matrix_widget)
        matrix_layout.setSpacing(2)
        matrix_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建13x13矩阵
        for row in range(13):
            for col in range(13):
                hand_label = HAND_LABELS[row][col]
                cell = RegretHeatmapCell(hand_label)
                cell.hovered.connect(self._on_cell_hovered)
                cell.clicked.connect(self._on_cell_clicked)
                matrix_layout.addWidget(cell, row, col)
                self._cells[hand_label] = cell
        
        layout.addWidget(matrix_widget, 1)
        
        # 图例
        legend_widget = self._create_legend()
        layout.addWidget(legend_widget)
        
        # 初始化为不可用状态
        self._set_unavailable()
    
    def _create_legend(self) -> QWidget:
        """创建图例控件。"""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border-top: 1px solid #ddd;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 颜色条
        color_bar = QLabel()
        color_bar.setFixedHeight(20)
        color_bar.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3264FF, stop:0.5 #FFFFFF, stop:1 #FF3232);
                border: 1px solid #999;
                border-radius: 3px;
            }
        """)
        layout.addWidget(color_bar)
        
        # 标签
        labels_layout = QGridLayout()
        labels_layout.addWidget(QLabel("低遗憾"), 0, 0, Qt.AlignmentFlag.AlignLeft)
        labels_layout.addWidget(QLabel("中等"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        labels_layout.addWidget(QLabel("高遗憾"), 0, 2, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(labels_layout)
        
        return widget
    
    def set_regret_result(self, result: RegretResult, has_board_cards: bool = True) -> None:
        """设置遗憾值计算结果。
        
        Args:
            result: 遗憾值计算结果
            has_board_cards: 是否有公共牌（翻后阶段需要）
        """
        self._regret_result = result
        self._is_available = True
        
        # 更新状态标签
        if not has_board_cards and result.node_state.stage.value != "preflop":
            # 翻后阶段没有公共牌，显示警告
            self._status_label.setText(
                f"⚠️ 请选择公共牌 | 遗憾值范围: {result.min_regret:.4f} ~ {result.max_regret:.4f}"
            )
            self._status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #e67e22;
                    padding: 5px;
                    font-weight: bold;
                }
            """)
        else:
            self._status_label.setText(
                f"遗憾值范围: {result.min_regret:.4f} ~ {result.max_regret:.4f}"
            )
            self._status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #666;
                    padding: 5px;
                }
            """)
        
        # 更新所有单元格
        for hand_label, cell in self._cells.items():
            if hand_label in result.hand_regrets:
                hand_regret = result.hand_regrets[hand_label]
                if hand_regret.is_blocked:
                    cell.set_blocked()
                else:
                    cell.set_regret(
                        hand_regret.average_total_regret,
                        hand_regret.average_regrets,
                        result.min_regret,
                        result.max_regret
                    )
            else:
                cell.set_unavailable()
    
    def _set_unavailable(self) -> None:
        """设置为不可用状态。"""
        self._is_available = False
        self._status_label.setText("遗憾网络不可用")
        
        for cell in self._cells.values():
            cell.set_unavailable()
    
    def set_unavailable(self) -> None:
        """公共方法：设置为不可用状态。"""
        self._set_unavailable()
    
    def _on_cell_hovered(self, hand_label: str) -> None:
        """处理单元格悬停事件。"""
        self.cell_hovered.emit(hand_label)
        
        # 更新状态标签显示具体遗憾值
        if self._regret_result and hand_label in self._regret_result.hand_regrets:
            hand_regret = self._regret_result.hand_regrets[hand_label]
            if hand_regret.is_blocked:
                self._status_label.setText(f"{hand_label}: 被公共牌阻挡")
            else:
                # 显示各动作的遗憾值
                regrets_str = " | ".join(
                    f"{name}:{val:+.2f}" 
                    for name, val in hand_regret.average_regrets.items()
                )
                self._status_label.setText(f"{hand_label}: {regrets_str}")
    
    def _on_cell_clicked(self, hand_label: str) -> None:
        """处理单元格点击事件。"""
        self.cell_clicked.emit(hand_label)
    
    def get_hand_regret(self, hand_label: str) -> Optional[HandRegret]:
        """获取指定手牌的遗憾值信息。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            手牌遗憾值，如果不存在则返回None
        """
        if self._regret_result and hand_label in self._regret_result.hand_regrets:
            return self._regret_result.hand_regrets[hand_label]
        return None
    
    def refresh(self) -> None:
        """刷新显示。"""
        self.update()
