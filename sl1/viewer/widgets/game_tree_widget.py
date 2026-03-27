"""游戏树控件模块。

本模块实现了游戏树的可视化控件，支持：
- 显示游戏树结构
- 展开/折叠节点
- 节点选择和导航
- 发出node_selected信号

需求引用:
- 需求 2.1: 显示从"Game Begin"开始的游戏树根节点
- 需求 2.2: 点击游戏阶段节点展开可用行动选项
- 需求 2.3: 选择行动更新当前节点
- 需求 2.5: 点击历史路径中的任意节点回退到该节点状态
"""

from typing import Optional, Dict, List, Callable
from PyQt6.QtWidgets import (
    QTreeWidget, 
    QTreeWidgetItem, 
    QWidget,
    QVBoxLayout,
    QHeaderView,
    QAbstractItemView,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor, QBrush, QIcon

from viewer.models import GameTreeNode, NodeType
from viewer.game_tree import GameTreeNavigator
from models.core import GameStage, ActionType


# 游戏阶段的中文名称
STAGE_NAMES = {
    GameStage.PREFLOP: "翻牌前",
    GameStage.FLOP: "翻牌",
    GameStage.TURN: "转牌",
    GameStage.RIVER: "河牌",
}

# 行动类型的中文名称
ACTION_NAMES = {
    ActionType.FOLD: "弃牌",
    ActionType.CHECK: "过牌",
    ActionType.CALL: "跟注",
    ActionType.RAISE: "加注",
    ActionType.RAISE_SMALL: "小加注",
    ActionType.RAISE_BIG: "大加注",
}

# 所有加注类型
RAISE_TYPES = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)

# 节点类型的图标颜色
NODE_COLORS = {
    NodeType.ROOT: QColor(100, 149, 237),      # 蓝色 - 根节点
    NodeType.CHANCE: QColor(255, 165, 0),      # 橙色 - 机会节点
    NodeType.PLAYER: QColor(50, 205, 50),      # 绿色 - 玩家节点
    NodeType.TERMINAL: QColor(220, 20, 60),    # 红色 - 终端节点
}


class GameTreeWidget(QTreeWidget):
    """游戏树控件 - 显示和交互游戏树的控件。
    
    该控件继承自QTreeWidget，提供以下功能：
    1. 显示游戏树的层级结构
    2. 支持展开/折叠节点
    3. 节点选择时发出信号
    4. 高亮显示当前路径
    
    Signals:
        node_selected(GameTreeNode): 节点被选中时发出
    
    Attributes:
        _navigator: 游戏树导航器
        _node_items: 节点ID到TreeWidgetItem的映射
        _current_node: 当前选中的节点
    """
    
    # 定义信号：节点被选中时发出
    node_selected = pyqtSignal(object)  # 发送GameTreeNode对象
    
    def __init__(
        self, 
        navigator: Optional[GameTreeNavigator] = None,
        parent: Optional[QWidget] = None
    ):
        """初始化游戏树控件。
        
        Args:
            navigator: 游戏树导航器，如果为None则创建默认导航器
            parent: 父控件
        """
        super().__init__(parent)
        
        # 初始化导航器
        self._navigator = navigator if navigator else GameTreeNavigator(max_raises_per_street=2)
        
        # 节点ID到TreeWidgetItem的映射
        self._node_items: Dict[str, QTreeWidgetItem] = {}
        
        # 当前选中的节点
        self._current_node: Optional[GameTreeNode] = None
        
        # 设置控件属性
        self._setup_widget()
        
        # 构建树形结构
        self._build_tree()
        
        # 连接信号
        self.itemClicked.connect(self._on_item_clicked)
        self.itemExpanded.connect(self._on_item_expanded)
        self.itemCollapsed.connect(self._on_item_collapsed)
    
    def _setup_widget(self) -> None:
        """设置控件的基本属性。"""
        # 设置列标题
        self.setHeaderLabels(["游戏树", "玩家", "底池"])
        
        # 设置列宽 - 第一列设置最小宽度
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 50)
        header.resizeSection(2, 60)
        header.setMinimumSectionSize(200)  # 第一列最小宽度，增加以显示完整行动名称
        
        # 设置选择模式
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        
        # 设置展开动画
        self.setAnimated(True)
        
        # 设置交替行颜色
        self.setAlternatingRowColors(True)
        
        # 设置缩进
        self.setIndentation(15)
        
        # 设置字体
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
        
        # 设置文本不截断
        self.setWordWrap(False)
    
    def _build_tree(self) -> None:
        """构建游戏树的可视化结构。"""
        # 清空现有内容
        self.clear()
        self._node_items.clear()
        
        # 获取根节点
        root_node = self._navigator.get_root()
        
        # 创建根节点项
        root_item = self._create_tree_item(root_node)
        self.addTopLevelItem(root_item)
        
        # 递归构建子节点
        self._build_children(root_node, root_item)
        
        # 展开根节点
        root_item.setExpanded(True)
        
        # 选中根节点
        self.setCurrentItem(root_item)
        self._current_node = root_node
    
    def _build_children(
        self, 
        parent_node: GameTreeNode, 
        parent_item: QTreeWidgetItem
    ) -> None:
        """递归构建子节点。
        
        Args:
            parent_node: 父节点
            parent_item: 父节点的TreeWidgetItem
        """
        for child_node in parent_node.children:
            child_item = self._create_tree_item(child_node)
            parent_item.addChild(child_item)
            
            # 递归构建子节点的子节点
            self._build_children(child_node, child_item)
    
    def _create_tree_item(self, node: GameTreeNode) -> QTreeWidgetItem:
        """为节点创建TreeWidgetItem。
        
        Args:
            node: 游戏树节点
            
        Returns:
            创建的TreeWidgetItem
        """
        item = QTreeWidgetItem()
        
        # 设置节点显示文本
        display_text = self._get_node_display_text(node)
        item.setText(0, display_text)
        
        # 设置玩家信息
        # 对于行动节点，显示执行该行动的玩家（父节点的玩家）
        # 对于决策节点，显示当前需要行动的玩家
        if node.action is not None and node.parent is not None and node.parent.player >= 0:
            # 这是一个行动节点，显示执行行动的玩家
            item.setText(1, f"P{node.parent.player + 1}")
        elif node.player >= 0:
            # 这是一个决策节点，显示当前玩家
            item.setText(1, f"P{node.player + 1}")
        else:
            item.setText(1, "-")
        
        # 设置底池信息
        item.setText(2, f"${node.pot}")
        
        # 设置节点颜色
        color = NODE_COLORS.get(node.node_type, QColor(128, 128, 128))
        item.setForeground(0, QBrush(color))
        
        # 设置工具提示
        tooltip = self._get_node_tooltip(node)
        item.setToolTip(0, tooltip)
        
        # 存储节点引用
        item.setData(0, Qt.ItemDataRole.UserRole, node)
        
        # 添加到映射
        self._node_items[node.node_id] = item
        
        return item
    
    def _get_node_display_text(self, node: GameTreeNode) -> str:
        """获取节点的显示文本。
        
        Args:
            node: 游戏树节点
            
        Returns:
            显示文本
        """
        # 根节点
        if node.node_type == NodeType.ROOT:
            return "[开始] 游戏开始"
        
        # 终端节点
        if node.node_type == NodeType.TERMINAL:
            if node.action is not None:
                action_name = ACTION_NAMES.get(
                    node.action.action_type, 
                    node.action.action_type.value
                )
                return f"[结束] {action_name}"
            return "[结束] 游戏结束"
        
        # 机会节点（发牌）
        if node.node_type == NodeType.CHANCE:
            stage_name = STAGE_NAMES.get(node.stage, str(node.stage.value))
            return f"[发牌] {stage_name}"
        
        # 玩家节点（行动）
        if node.action is not None:
            action_name = ACTION_NAMES.get(
                node.action.action_type, 
                node.action.action_type.value
            )
            if node.action.action_type in RAISE_TYPES:
                return f"[行动] {action_name} ${node.action.amount}"
            return f"[行动] {action_name}"
        
        # 默认显示阶段（玩家决策点）
        stage_name = STAGE_NAMES.get(node.stage, str(node.stage.value))
        return f"[决策] {stage_name} - 玩家{node.player + 1}"
    
    def _get_node_tooltip(self, node: GameTreeNode) -> str:
        """获取节点的工具提示文本。
        
        Args:
            node: 游戏树节点
            
        Returns:
            工具提示文本
        """
        lines = []
        
        # 节点ID
        lines.append(f"节点ID: {node.node_id}")
        
        # 游戏阶段
        stage_name = STAGE_NAMES.get(node.stage, str(node.stage.value))
        lines.append(f"阶段: {stage_name}")
        
        # 当前玩家
        if node.player >= 0:
            lines.append(f"当前玩家: 玩家{node.player + 1}")
        
        # 底池
        lines.append(f"底池: ${node.pot}")
        
        # 筹码
        lines.append(f"筹码: P1=${node.stacks[0]}, P2=${node.stacks[1]}")
        
        # 公共牌
        if node.board_cards:
            board_str = " ".join(str(card) for card in node.board_cards)
            lines.append(f"公共牌: {board_str}")
        
        # 行动历史
        if node.action_history:
            history_str = " → ".join(
                ACTION_NAMES.get(a.action_type, a.action_type.value)
                for a in node.action_history[-5:]  # 只显示最近5个行动
            )
            if len(node.action_history) > 5:
                history_str = "... → " + history_str
            lines.append(f"行动历史: {history_str}")
        
        return "\n".join(lines)
    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """处理节点点击事件。
        
        Args:
            item: 被点击的TreeWidgetItem
            column: 被点击的列
        """
        # 获取节点数据
        node = item.data(0, Qt.ItemDataRole.UserRole)
        
        if node is None:
            return
        
        # 更新当前节点
        self._current_node = node
        
        # 导航到该节点
        self._navigator.navigate_to(node)
        
        # 高亮显示路径
        self._highlight_path(node)
        
        # 发出信号
        self.node_selected.emit(node)
    
    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        """处理节点展开事件。
        
        Args:
            item: 被展开的TreeWidgetItem
        """
        # 可以在这里添加延迟加载逻辑
        pass
    
    def _on_item_collapsed(self, item: QTreeWidgetItem) -> None:
        """处理节点折叠事件。
        
        Args:
            item: 被折叠的TreeWidgetItem
        """
        pass
    
    def _highlight_path(self, node: GameTreeNode) -> None:
        """高亮显示从根节点到指定节点的路径。
        
        Args:
            node: 目标节点
        """
        # 重置所有节点的背景色
        for item in self._node_items.values():
            item.setBackground(0, QBrush())
            item.setBackground(1, QBrush())
            item.setBackground(2, QBrush())
        
        # 获取路径
        path = node.get_path_to_root()
        
        # 高亮路径上的节点
        highlight_color = QColor(255, 255, 200)  # 淡黄色
        for path_node in path:
            if path_node.node_id in self._node_items:
                item = self._node_items[path_node.node_id]
                item.setBackground(0, QBrush(highlight_color))
                item.setBackground(1, QBrush(highlight_color))
                item.setBackground(2, QBrush(highlight_color))
        
        # 确保当前节点可见
        if node.node_id in self._node_items:
            current_item = self._node_items[node.node_id]
            self.scrollToItem(current_item)
    
    # ========================================================================
    # 公共方法
    # ========================================================================
    
    def set_navigator(self, navigator: GameTreeNavigator) -> None:
        """设置游戏树导航器。
        
        Args:
            navigator: 新的导航器
        """
        self._navigator = navigator
        self._build_tree()
    
    def get_navigator(self) -> GameTreeNavigator:
        """获取当前的游戏树导航器。
        
        Returns:
            游戏树导航器
        """
        return self._navigator
    
    def get_current_node(self) -> Optional[GameTreeNode]:
        """获取当前选中的节点。
        
        Returns:
            当前节点，如果没有选中则返回None
        """
        return self._current_node
    
    def select_node(self, node: GameTreeNode) -> bool:
        """选中指定的节点。
        
        Args:
            node: 要选中的节点
            
        Returns:
            是否选中成功
        """
        if node.node_id not in self._node_items:
            return False
        
        item = self._node_items[node.node_id]
        
        # 展开父节点以确保可见
        self._expand_to_node(node)
        
        # 选中节点
        self.setCurrentItem(item)
        self._current_node = node
        
        # 高亮路径
        self._highlight_path(node)
        
        # 发出信号
        self.node_selected.emit(node)
        
        return True
    
    def select_node_by_id(self, node_id: str) -> bool:
        """通过节点ID选中节点。
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否选中成功
        """
        node = self._navigator.get_node_by_id(node_id)
        if node is None:
            return False
        return self.select_node(node)
    
    def _expand_to_node(self, node: GameTreeNode) -> None:
        """展开从根节点到指定节点的所有父节点。
        
        Args:
            node: 目标节点
        """
        path = node.get_path_to_root()
        
        for path_node in path[:-1]:  # 不包括目标节点本身
            if path_node.node_id in self._node_items:
                item = self._node_items[path_node.node_id]
                item.setExpanded(True)
    
    def expand_all_nodes(self) -> None:
        """展开所有节点。"""
        self.expandAll()
    
    def collapse_all_nodes(self) -> None:
        """折叠所有节点（保留根节点展开）。"""
        self.collapseAll()
        
        # 保持根节点展开
        root_node = self._navigator.get_root()
        if root_node.node_id in self._node_items:
            self._node_items[root_node.node_id].setExpanded(True)
    
    def refresh(self) -> None:
        """刷新游戏树显示。"""
        # 保存当前选中的节点ID
        current_node_id = self._current_node.node_id if self._current_node else None
        
        # 重建树
        self._build_tree()
        
        # 恢复选中状态
        if current_node_id:
            self.select_node_by_id(current_node_id)
    
    def add_node(
        self, 
        parent_node: GameTreeNode, 
        child_node: GameTreeNode
    ) -> bool:
        """添加新节点到树中。
        
        Args:
            parent_node: 父节点
            child_node: 要添加的子节点
            
        Returns:
            是否添加成功
        """
        if parent_node.node_id not in self._node_items:
            return False
        
        parent_item = self._node_items[parent_node.node_id]
        child_item = self._create_tree_item(child_node)
        parent_item.addChild(child_item)
        
        return True
    
    def get_path_display(self) -> str:
        """获取当前路径的显示文本。
        
        Returns:
            路径显示文本，如 "游戏开始 → 翻牌前 → 跟注 → 加注"
        """
        if self._current_node is None:
            return ""
        
        path = self._current_node.get_path_to_root()
        path_texts = []
        
        for node in path:
            text = self._get_node_display_text(node)
            # 移除emoji前缀
            text = text.lstrip("🎮🎴🃏💰🏁📍 ")
            path_texts.append(text)
        
        return " → ".join(path_texts)
