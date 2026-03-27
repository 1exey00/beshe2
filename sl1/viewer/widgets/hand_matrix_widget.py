"""手牌范围矩阵控件模块。

本模块实现了13x13手牌范围矩阵的可视化控件，支持：
- 绘制13x13矩阵，显示颜色编码
- 鼠标悬停显示手牌信息
- 点击选择手牌
- 发出cell_hovered和cell_clicked信号

需求引用:
- 需求 3.1: 选择决策节点时显示13x13的手牌范围矩阵
- 需求 3.2: 使用颜色编码表示主要行动
- 需求 4.1: 鼠标悬停显示详细策略概率
"""

from typing import Optional, Dict, Set, Tuple
from PyQt6.QtWidgets import (
    QWidget,
    QToolTip,
    QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt, QRect, QPoint, QSize
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QFont,
    QPen,
    QBrush,
    QMouseEvent,
    QPaintEvent,
    QResizeEvent,
    QFontMetrics,
)

from viewer.hand_range import HAND_LABELS_MATRIX, RANKS_ORDER
from viewer.color_mapper import StrategyColorMapper, Color


class HandRangeMatrixWidget(QWidget):
    """手牌范围矩阵控件 - 显示13x13手牌范围矩阵。
    
    该控件提供以下功能：
    1. 绘制13x13的手牌范围矩阵
    2. 使用颜色编码显示策略分布
    3. 鼠标悬停时显示手牌标签和策略信息
    4. 点击时发出信号
    
    Signals:
        cell_hovered(str): 鼠标悬停在单元格上时发出手牌标签
        cell_clicked(str): 单元格被点击时发出手牌标签
    
    Attributes:
        _strategies: 手牌策略字典 {手牌标签: {行动: 概率}}
        _disabled_hands: 禁用的手牌集合（与公共牌冲突）
        _color_mapper: 颜色映射器
        _hovered_cell: 当前悬停的单元格位置
        _selected_cell: 当前选中的单元格位置
    """
    
    # 定义信号
    cell_hovered = pyqtSignal(str)   # 发送手牌标签
    cell_clicked = pyqtSignal(str)   # 发送手牌标签
    
    # 矩阵尺寸
    MATRIX_SIZE = 13
    
    # 最小单元格大小
    MIN_CELL_SIZE = 30
    
    # 边距
    MARGIN = 5
    HEADER_SIZE = 25  # 行/列标题大小
    
    def __init__(self, parent: Optional[QWidget] = None):
        """初始化手牌矩阵控件。
        
        Args:
            parent: 父控件
        """
        super().__init__(parent)
        
        # 策略数据 {手牌标签: {行动: 概率}}
        self._strategies: Dict[str, Dict[str, float]] = {}
        
        # 禁用的手牌（与公共牌冲突）
        self._disabled_hands: Set[str] = set()
        
        # 颜色映射器
        self._color_mapper = StrategyColorMapper()
        
        # 当前悬停的单元格 (row, col)
        self._hovered_cell: Optional[Tuple[int, int]] = None
        
        # 当前选中的单元格 (row, col)
        self._selected_cell: Optional[Tuple[int, int]] = None
        
        # 单元格大小（动态计算）
        self._cell_size = self.MIN_CELL_SIZE
        
        # 设置控件属性
        self._setup_widget()
    
    def _setup_widget(self) -> None:
        """设置控件的基本属性。"""
        # 启用鼠标追踪（用于悬停效果）
        self.setMouseTracking(True)
        
        # 设置大小策略
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # 设置最小大小
        min_size = (
            self.MATRIX_SIZE * self.MIN_CELL_SIZE + 
            self.HEADER_SIZE + 
            2 * self.MARGIN
        )
        self.setMinimumSize(min_size, min_size)
        
        # 设置背景色
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setPalette(palette)
    
    def sizeHint(self) -> QSize:
        """返回控件的建议大小。"""
        preferred_size = (
            self.MATRIX_SIZE * 40 +  # 40像素单元格
            self.HEADER_SIZE + 
            2 * self.MARGIN
        )
        return QSize(preferred_size, preferred_size)
    
    def resizeEvent(self, event: QResizeEvent) -> None:
        """处理窗口大小改变事件。"""
        super().resizeEvent(event)
        self._calculate_cell_size()
    
    def _calculate_cell_size(self) -> None:
        """计算单元格大小。"""
        # 可用空间
        available_width = self.width() - self.HEADER_SIZE - 2 * self.MARGIN
        available_height = self.height() - self.HEADER_SIZE - 2 * self.MARGIN
        
        # 计算单元格大小（取较小值以保持正方形）
        self._cell_size = max(
            self.MIN_CELL_SIZE,
            min(available_width, available_height) // self.MATRIX_SIZE
        )
    
    def _get_cell_rect(self, row: int, col: int) -> QRect:
        """获取单元格的矩形区域。
        
        Args:
            row: 行索引 (0-12)
            col: 列索引 (0-12)
            
        Returns:
            单元格的QRect
        """
        x = self.MARGIN + self.HEADER_SIZE + col * self._cell_size
        y = self.MARGIN + self.HEADER_SIZE + row * self._cell_size
        return QRect(x, y, self._cell_size, self._cell_size)
    
    def _get_cell_at_pos(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        """获取指定位置的单元格索引。
        
        Args:
            pos: 鼠标位置
            
        Returns:
            (row, col) 元组，如果不在单元格内则返回None
        """
        # 计算相对于矩阵起点的位置
        x = pos.x() - self.MARGIN - self.HEADER_SIZE
        y = pos.y() - self.MARGIN - self.HEADER_SIZE
        
        if x < 0 or y < 0:
            return None
        
        col = x // self._cell_size
        row = y // self._cell_size
        
        if 0 <= row < self.MATRIX_SIZE and 0 <= col < self.MATRIX_SIZE:
            return (row, col)
        
        return None
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """绘制控件。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制行/列标题
        self._draw_headers(painter)
        
        # 绘制矩阵单元格
        self._draw_cells(painter)
        
        # 绘制悬停高亮
        if self._hovered_cell is not None:
            self._draw_cell_highlight(painter, self._hovered_cell, QColor(0, 0, 0, 50))
        
        # 绘制选中高亮
        if self._selected_cell is not None:
            self._draw_cell_highlight(painter, self._selected_cell, QColor(0, 0, 255, 100))
    
    def _draw_headers(self, painter: QPainter) -> None:
        """绘制行/列标题。"""
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(50, 50, 50)))
        
        # 绘制列标题（A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2）
        for col, rank in enumerate(RANKS_ORDER):
            x = self.MARGIN + self.HEADER_SIZE + col * self._cell_size
            y = self.MARGIN
            rect = QRect(x, y, self._cell_size, self.HEADER_SIZE)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, rank)
        
        # 绘制行标题
        for row, rank in enumerate(RANKS_ORDER):
            x = self.MARGIN
            y = self.MARGIN + self.HEADER_SIZE + row * self._cell_size
            rect = QRect(x, y, self.HEADER_SIZE, self._cell_size)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, rank)
    
    def _draw_cells(self, painter: QPainter) -> None:
        """绘制所有单元格。"""
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        
        for row in range(self.MATRIX_SIZE):
            for col in range(self.MATRIX_SIZE):
                self._draw_cell(painter, row, col)
    
    def _draw_cell(self, painter: QPainter, row: int, col: int) -> None:
        """绘制单个单元格。
        
        使用条状组合显示策略分布（需求6.1-6.6）：
        - 水平条状组合显示各动作的概率比例
        - 条状宽度按概率比例分配
        - 概率为0的动作不显示
        
        Args:
            painter: QPainter对象
            row: 行索引
            col: 列索引
        """
        rect = self._get_cell_rect(row, col)
        hand_label = HAND_LABELS_MATRIX[row][col]
        
        # 获取单元格颜色/条状段
        if hand_label in self._disabled_hands:
            # 禁用的手牌（与公共牌冲突）- 使用单一颜色
            color = self._color_mapper.get_disabled_color()
            bg_color = QColor(color.r, color.g, color.b, color.a)
            painter.fillRect(rect, bg_color)
        elif hand_label in self._strategies:
            # 有策略数据 - 使用条状组合显示
            strategy = self._strategies[hand_label]
            self._draw_bar_segments(painter, rect, strategy)
        else:
            # 无策略数据 - 使用背景色
            color = self._color_mapper.get_background_color()
            bg_color = QColor(color.r, color.g, color.b, color.a)
            painter.fillRect(rect, bg_color)
        
        # 绘制边框
        border_color = QColor(200, 200, 200)
        painter.setPen(QPen(border_color, 1))
        painter.drawRect(rect)
        
        # 绘制手牌标签
        # 计算背景的平均颜色用于确定文字颜色
        if hand_label in self._disabled_hands:
            label_bg_color = self._color_mapper.get_disabled_color()
        elif hand_label in self._strategies:
            # 使用混合颜色来确定文字颜色
            label_bg_color = self._color_mapper.get_cell_color(self._strategies[hand_label])
        else:
            label_bg_color = self._color_mapper.get_background_color()
        
        self._draw_cell_label(painter, rect, hand_label, label_bg_color)
    
    def _draw_bar_segments(
        self, 
        painter: QPainter, 
        rect: QRect, 
        strategy: Dict[str, float]
    ) -> None:
        """绘制条状组合显示策略分布。
        
        根据需求6.1-6.6：
        - 使用水平条状组合显示各动作的概率比例
        - 条状宽度按概率比例分配
        - 概率为0的动作不显示
        
        Args:
            painter: QPainter对象
            rect: 单元格矩形区域
            strategy: 策略概率字典 {动作: 概率}
        """
        # 获取条状段列表
        segments = self._color_mapper.get_bar_segments(strategy)
        
        if not segments:
            # 如果没有有效的条状段，使用背景色
            color = self._color_mapper.get_background_color()
            bg_color = QColor(color.r, color.g, color.b, color.a)
            painter.fillRect(rect, bg_color)
            return
        
        # 计算每个条状段的位置和宽度
        x_start = rect.x()
        y = rect.y()
        total_width = rect.width()
        height = rect.height()
        
        current_x = x_start
        
        for segment in segments:
            # 计算条状宽度（按概率比例）
            segment_width = int(total_width * segment.width_ratio)
            
            # 确保最后一个段填满剩余空间（处理舍入误差）
            if segment == segments[-1]:
                segment_width = x_start + total_width - current_x
            
            # 跳过宽度为0的段
            if segment_width <= 0:
                continue
            
            # 绘制条状段
            segment_rect = QRect(current_x, y, segment_width, height)
            segment_color = QColor(segment.color[0], segment.color[1], segment.color[2])
            painter.fillRect(segment_rect, segment_color)
            
            current_x += segment_width
    
    def _draw_cell_label(
        self, 
        painter: QPainter, 
        rect: QRect, 
        label: str,
        bg_color: Color
    ) -> None:
        """绘制单元格标签。
        
        Args:
            painter: QPainter对象
            rect: 单元格矩形
            label: 手牌标签
            bg_color: 背景颜色（用于计算文字颜色）
        """
        # 根据背景亮度选择文字颜色
        brightness = (bg_color.r * 299 + bg_color.g * 587 + bg_color.b * 114) / 1000
        text_color = QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)
        
        painter.setPen(QPen(text_color))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
    
    def _draw_cell_highlight(
        self, 
        painter: QPainter, 
        cell: Tuple[int, int],
        color: QColor
    ) -> None:
        """绘制单元格高亮效果。
        
        Args:
            painter: QPainter对象
            cell: (row, col) 单元格位置
            color: 高亮颜色
        """
        row, col = cell
        rect = self._get_cell_rect(row, col)
        
        # 绘制半透明覆盖层
        painter.fillRect(rect, color)
        
        # 绘制加粗边框
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawRect(rect)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """处理鼠标移动事件。"""
        cell = self._get_cell_at_pos(event.pos())
        
        if cell != self._hovered_cell:
            self._hovered_cell = cell
            self.update()  # 触发重绘
            
            if cell is not None:
                row, col = cell
                hand_label = HAND_LABELS_MATRIX[row][col]
                
                # 发出信号
                self.cell_hovered.emit(hand_label)
                
                # 显示工具提示
                self._show_tooltip(event.globalPosition().toPoint(), hand_label)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """处理鼠标点击事件。"""
        if event.button() == Qt.MouseButton.LeftButton:
            cell = self._get_cell_at_pos(event.pos())
            
            if cell is not None:
                self._selected_cell = cell
                self.update()  # 触发重绘
                
                row, col = cell
                hand_label = HAND_LABELS_MATRIX[row][col]
                
                # 发出信号
                self.cell_clicked.emit(hand_label)
    
    def leaveEvent(self, event) -> None:
        """处理鼠标离开事件。"""
        self._hovered_cell = None
        self.update()
        QToolTip.hideText()
    
    def _show_tooltip(self, global_pos: QPoint, hand_label: str) -> None:
        """显示工具提示。
        
        Args:
            global_pos: 全局鼠标位置
            hand_label: 手牌标签
        """
        # 构建提示文本
        lines = [f"<b>{hand_label}</b>"]
        
        # 添加手牌类型说明
        if len(hand_label) == 2:
            lines.append("对子")
        elif hand_label.endswith('s'):
            lines.append("同花")
        else:
            lines.append("非同花")
        
        # 添加策略信息
        if hand_label in self._disabled_hands:
            lines.append("<i>与公共牌冲突</i>")
        elif hand_label in self._strategies:
            strategy = self._strategies[hand_label]
            lines.append("<br><b>策略:</b>")
            for action, prob in sorted(
                strategy.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                if prob > 0.001:  # 只显示概率大于0.1%的行动
                    lines.append(f"  {action}: {prob*100:.1f}%")
        else:
            lines.append("<i>无策略数据</i>")
        
        tooltip_text = "<br>".join(lines)
        QToolTip.showText(global_pos, tooltip_text, self)
    
    # ========================================================================
    # 公共方法
    # ========================================================================
    
    def set_strategies(self, strategies: Dict[str, Dict[str, float]]) -> None:
        """设置手牌策略数据。
        
        Args:
            strategies: 策略字典 {手牌标签: {行动: 概率}}
        """
        self._strategies = strategies.copy()
        self.update()
    
    def get_strategies(self) -> Dict[str, Dict[str, float]]:
        """获取当前的策略数据。
        
        Returns:
            策略字典
        """
        return self._strategies.copy()
    
    def set_disabled_hands(self, hands: Set[str]) -> None:
        """设置禁用的手牌（与公共牌冲突）。
        
        Args:
            hands: 禁用的手牌标签集合
        """
        self._disabled_hands = hands.copy()
        self.update()
    
    def get_disabled_hands(self) -> Set[str]:
        """获取禁用的手牌集合。
        
        Returns:
            禁用的手牌标签集合
        """
        return self._disabled_hands.copy()
    
    def clear_strategies(self) -> None:
        """清空策略数据。"""
        self._strategies.clear()
        self.update()
    
    def clear_disabled_hands(self) -> None:
        """清空禁用的手牌。"""
        self._disabled_hands.clear()
        self.update()
    
    def clear_selection(self) -> None:
        """清除选中状态。"""
        self._selected_cell = None
        self.update()
    
    def select_hand(self, hand_label: str) -> bool:
        """选中指定的手牌。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            是否选中成功
        """
        for row in range(self.MATRIX_SIZE):
            for col in range(self.MATRIX_SIZE):
                if HAND_LABELS_MATRIX[row][col] == hand_label:
                    self._selected_cell = (row, col)
                    self.update()
                    return True
        return False
    
    def get_selected_hand(self) -> Optional[str]:
        """获取当前选中的手牌标签。
        
        Returns:
            手牌标签，如果没有选中则返回None
        """
        if self._selected_cell is None:
            return None
        row, col = self._selected_cell
        return HAND_LABELS_MATRIX[row][col]
    
    def get_hovered_hand(self) -> Optional[str]:
        """获取当前悬停的手牌标签。
        
        Returns:
            手牌标签，如果没有悬停则返回None
        """
        if self._hovered_cell is None:
            return None
        row, col = self._hovered_cell
        return HAND_LABELS_MATRIX[row][col]
    
    def set_color_mapper(self, mapper: StrategyColorMapper) -> None:
        """设置颜色映射器。
        
        Args:
            mapper: 颜色映射器
        """
        self._color_mapper = mapper
        self.update()
    
    def refresh(self) -> None:
        """刷新显示。"""
        self.update()
