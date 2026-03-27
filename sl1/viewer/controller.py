"""策略查看器主控制器模块。

本模块实现了策略查看器的主控制器，协调各组件并提供统一的API接口：
- 模型加载和管理
- 游戏树导航
- 策略计算和显示
- 公共牌设置
- 数据导出

需求引用:
- 需求 1.1: 加载检查点文件
- 需求 2.3: 选择行动更新节点
- 需求 6.2: 设置公共牌更新策略
- 需求 6.4: 公共牌改变自动重新计算
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
import json

from models.core import Card, GameStage
from viewer.models import GameTreeNode, HandStrategy, NodeState, ActionConfig
from viewer.model_loader import ModelLoader, ModelMetadata
from viewer.game_tree import GameTreeNavigator
from viewer.hand_range import HandRangeCalculator, HAND_LABELS_MATRIX
from viewer.strategy_calculator import StrategyCalculator, StrategyResult
from viewer.color_mapper import StrategyColorMapper, Color
from viewer.value_calculator import ValueCalculator, ValueResult, HandValue
from viewer.regret_calculator import RegretCalculator, RegretResult, HandRegret
from analysis.strategy_analyzer import StrategyAnalyzer


@dataclass
class ViewerState:
    """查看器状态信息。
    
    Attributes:
        is_model_loaded: 模型是否已加载
        current_node_id: 当前节点ID
        board_cards: 当前公共牌
        selected_hand: 当前选中的手牌标签
    """
    is_model_loaded: bool = False
    current_node_id: str = "root"
    board_cards: List[Card] = None
    selected_hand: Optional[str] = None
    
    def __post_init__(self):
        if self.board_cards is None:
            self.board_cards = []


class StrategyViewerController:
    """策略查看器主控制器 - 协调各组件，提供统一的API接口。
    
    该控制器是策略查看器的核心，负责：
    1. 管理模型加载和卸载
    2. 协调游戏树导航
    3. 计算和缓存策略数据
    4. 处理公共牌设置
    5. 提供数据导出功能
    
    Attributes:
        model_loader: 模型加载器
        game_tree: 游戏树导航器
        strategy_calculator: 策略计算器
        color_mapper: 颜色映射器
        hand_range_calculator: 手牌范围计算器
    """
    
    def __init__(self, device: str = "cpu"):
        """初始化策略查看器控制器。
        
        Args:
            device: 计算设备（cpu或cuda）
        """
        self._device = device
        
        # 初始化各组件
        self._model_loader = ModelLoader(device=device)
        self._game_tree = GameTreeNavigator(max_raises_per_street=2)
        self._strategy_analyzer: Optional[StrategyAnalyzer] = None
        self._strategy_calculator: Optional[StrategyCalculator] = None
        self._value_calculator = ValueCalculator()
        self._regret_calculator = RegretCalculator()
        self._color_mapper = StrategyColorMapper()
        self._hand_range_calculator = HandRangeCalculator()
        
        # 状态管理
        self._state = ViewerState()
        self._cached_strategy: Optional[StrategyResult] = None
        self._cached_value: Optional[ValueResult] = None
        self._cached_regret: Optional[RegretResult] = None
        
        # 事件回调
        self._on_model_loaded: Optional[Callable[[ModelMetadata], None]] = None
        self._on_node_changed: Optional[Callable[[GameTreeNode], None]] = None
        self._on_strategy_updated: Optional[Callable[[StrategyResult], None]] = None
        self._on_board_changed: Optional[Callable[[List[Card]], None]] = None
        self._on_value_updated: Optional[Callable[[ValueResult], None]] = None
        self._on_regret_updated: Optional[Callable[[RegretResult], None]] = None
    
    # ========================================================================
    # 模型加载相关方法
    # ========================================================================
    
    def load_model(self, checkpoint_path: Union[str, Path]) -> ModelMetadata:
        """加载模型检查点。
        
        加载模型后会自动检测动作配置，并将其传递给策略计算器和颜色映射器。
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            模型元数据信息
            
        Raises:
            CheckpointNotFoundError: 检查点文件不存在
            CheckpointCorruptedError: 检查点文件损坏
            ModelLoadError: 模型加载失败
            
        需求引用:
        - 需求 1.4: 动作维度检测完成后在元数据中显示检测到的动作数量
        - 需求 2.1: 策略计算器使用从模型获取的动作列表
        """
        # 使用ModelLoader加载模型
        metadata = self._model_loader.load(checkpoint_path)
        
        # 获取动作配置（ModelLoader在加载时已自动检测）
        action_config = self._model_loader.action_config
        
        # 初始化策略分析器
        self._strategy_analyzer = StrategyAnalyzer(device=self._device)
        self._strategy_analyzer.load_model(checkpoint_path)
        
        # 如果策略分析器支持设置动作配置，则设置
        if hasattr(self._strategy_analyzer, 'set_action_config') and action_config:
            self._strategy_analyzer.set_action_config(action_config)
        
        # 初始化策略计算器，传入动作配置
        self._strategy_calculator = StrategyCalculator(
            strategy_analyzer=self._strategy_analyzer,
            action_config=action_config
        )
        
        # 初始化价值计算器（如果有价值网络 - 旧格式）
        if self._model_loader.has_value_network:
            self._value_calculator.set_value_network(self._model_loader.value_network)
        else:
            self._value_calculator.set_value_network(None)
        
        # 初始化遗憾值计算器（如果有遗憾网络 - Deep CFR格式）
        if self._model_loader.has_regret_network:
            self._regret_calculator.set_regret_network(self._model_loader.regret_network)
            # 设置动作配置
            if action_config:
                self._regret_calculator.set_action_config(
                    action_config.action_names, 
                    action_config.action_dim
                )
        else:
            self._regret_calculator.set_regret_network(None)
        
        # 更新状态
        self._state.is_model_loaded = True
        
        # 清除缓存
        self._cached_strategy = None
        self._cached_value = None
        self._cached_regret = None
        
        # 触发回调
        if self._on_model_loaded:
            self._on_model_loaded(metadata)
        
        # 自动计算当前节点的策略
        self._update_strategy()
        
        return metadata
    
    def unload_model(self) -> None:
        """卸载当前加载的模型。"""
        self._model_loader.unload()
        self._strategy_analyzer = None
        self._strategy_calculator = None
        self._value_calculator.set_value_network(None)
        self._regret_calculator.set_regret_network(None)
        self._state.is_model_loaded = False
        self._cached_strategy = None
        self._cached_value = None
        self._cached_regret = None
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载。
        
        Returns:
            模型是否已加载
        """
        return self._state.is_model_loaded
    
    def get_model_metadata(self) -> Optional[ModelMetadata]:
        """获取当前加载模型的元数据。
        
        Returns:
            模型元数据，如果未加载则返回None
        """
        return self._model_loader.metadata
    
    def get_action_config(self) -> Optional[ActionConfig]:
        """获取当前加载模型的动作配置。
        
        动作配置包含模型支持的动作名称列表和动作维度。
        如果模型未加载，返回None。
        
        Returns:
            动作配置对象，如果未加载模型则返回None
            
        需求引用:
        - 需求 1.4: 动作维度检测完成后在元数据中显示检测到的动作数量
        - 需求 2.1: 策略计算器使用从模型获取的动作列表
        """
        return self._model_loader.action_config
    
    # ========================================================================
    # 游戏树导航相关方法
    # ========================================================================
    
    def get_root_node(self) -> GameTreeNode:
        """获取游戏树根节点。
        
        Returns:
            根节点
        """
        return self._game_tree.get_root()
    
    def get_current_node(self) -> GameTreeNode:
        """获取当前选中的节点。
        
        Returns:
            当前节点
        """
        return self._game_tree.get_current_node()
    
    def get_children(self, node: Optional[GameTreeNode] = None) -> List[GameTreeNode]:
        """获取指定节点的子节点。
        
        Args:
            node: 目标节点，如果为None则使用当前节点
            
        Returns:
            子节点列表
        """
        return self._game_tree.get_children(node)
    
    def navigate_to(self, node: GameTreeNode) -> bool:
        """导航到指定节点。
        
        导航成功后会自动更新策略显示。
        
        Args:
            node: 目标节点
            
        Returns:
            是否导航成功
        """
        success = self._game_tree.navigate_to(node)
        
        if success:
            self._state.current_node_id = node.node_id
            
            # 触发回调
            if self._on_node_changed:
                self._on_node_changed(node)
            
            # 更新策略
            self._update_strategy()
        
        return success
    
    def navigate_to_by_id(self, node_id: str) -> bool:
        """通过节点ID导航。
        
        Args:
            node_id: 目标节点ID
            
        Returns:
            是否导航成功
        """
        success = self._game_tree.navigate_to_by_id(node_id)
        
        if success:
            self._state.current_node_id = node_id
            node = self._game_tree.get_current_node()
            
            if self._on_node_changed:
                self._on_node_changed(node)
            
            self._update_strategy()
        
        return success
    
    def navigate_by_action(self, action_name: str) -> bool:
        """通过行动名称导航到子节点。
        
        Args:
            action_name: 行动名称
            
        Returns:
            是否导航成功
        """
        success = self._game_tree.navigate_by_action(action_name)
        
        if success:
            node = self._game_tree.get_current_node()
            self._state.current_node_id = node.node_id
            
            if self._on_node_changed:
                self._on_node_changed(node)
            
            self._update_strategy()
        
        return success
    
    def navigate_to_parent(self) -> bool:
        """导航到父节点。
        
        Returns:
            是否导航成功
        """
        success = self._game_tree.navigate_to_parent()
        
        if success:
            node = self._game_tree.get_current_node()
            self._state.current_node_id = node.node_id
            
            if self._on_node_changed:
                self._on_node_changed(node)
            
            self._update_strategy()
        
        return success
    
    def get_path_to_root(self) -> List[GameTreeNode]:
        """获取从根节点到当前节点的路径。
        
        Returns:
            节点路径列表
        """
        return self._game_tree.get_path_to_root()
    
    def get_available_actions(self) -> List[str]:
        """获取当前节点的可用行动。
        
        Returns:
            可用行动名称列表
        """
        return self._game_tree.get_available_actions()
    
    def reset_to_root(self) -> None:
        """重置导航到根节点。"""
        self._game_tree.reset_to_root()
        self._state.current_node_id = "root"
        
        node = self._game_tree.get_current_node()
        if self._on_node_changed:
            self._on_node_changed(node)
        
        self._update_strategy()
    
    # ========================================================================
    # 公共牌设置相关方法
    # ========================================================================
    
    def set_board_cards(self, cards: List[Card]) -> None:
        """设置公共牌。
        
        设置后会自动重新计算策略。
        
        Args:
            cards: 公共牌列表
        """
        self._state.board_cards = cards.copy()
        
        # 触发回调
        if self._on_board_changed:
            self._on_board_changed(cards)
        
        # 重新计算策略
        self._update_strategy()
    
    def get_board_cards(self) -> List[Card]:
        """获取当前公共牌。
        
        Returns:
            公共牌列表
        """
        return self._state.board_cards.copy()
    
    def clear_board_cards(self) -> None:
        """清除公共牌。"""
        self.set_board_cards([])
    
    def add_board_card(self, card: Card) -> bool:
        """添加一张公共牌。
        
        Args:
            card: 要添加的牌
            
        Returns:
            是否添加成功（最多5张）
        """
        if len(self._state.board_cards) >= 5:
            return False
        
        # 检查是否重复
        for existing in self._state.board_cards:
            if existing.rank == card.rank and existing.suit == card.suit:
                return False
        
        new_board = self._state.board_cards + [card]
        self.set_board_cards(new_board)
        return True
    
    def remove_board_card(self, index: int) -> bool:
        """移除指定位置的公共牌。
        
        Args:
            index: 要移除的牌的索引
            
        Returns:
            是否移除成功
        """
        if index < 0 or index >= len(self._state.board_cards):
            return False
        
        new_board = self._state.board_cards.copy()
        new_board.pop(index)
        self.set_board_cards(new_board)
        return True
    
    # ========================================================================
    # 策略查询相关方法
    # ========================================================================
    
    def get_hand_strategy(self, hand_label: str) -> Optional[HandStrategy]:
        """获取指定手牌的策略。
        
        Args:
            hand_label: 手牌标签（如"AKs"）
            
        Returns:
            手牌策略，如果未计算则返回None
        """
        if self._cached_strategy is None:
            return None
        
        return self._cached_strategy.hand_strategies.get(hand_label)
    
    def get_all_hand_strategies(self) -> Dict[str, HandStrategy]:
        """获取所有手牌的策略。
        
        Returns:
            手牌策略字典
        """
        if self._cached_strategy is None:
            return {}
        
        return self._cached_strategy.hand_strategies.copy()
    
    def get_hand_color(self, hand_label: str) -> Color:
        """获取指定手牌的显示颜色。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            颜色对象
        """
        strategy = self.get_hand_strategy(hand_label)
        
        if strategy is None:
            return self._color_mapper.get_background_color()
        
        # 检查是否有有效组合
        if not strategy.combinations:
            return self._color_mapper.get_disabled_color()
        
        return self._color_mapper.get_cell_color(strategy.action_probabilities)
    
    def get_matrix_colors(self) -> List[List[Color]]:
        """获取13x13手牌矩阵的颜色。
        
        Returns:
            13x13的颜色矩阵
        """
        colors = []
        for row in HAND_LABELS_MATRIX:
            row_colors = []
            for label in row:
                row_colors.append(self.get_hand_color(label))
            colors.append(row_colors)
        return colors
    
    def get_node_state(self) -> Optional[NodeState]:
        """获取当前节点的状态信息。
        
        Returns:
            节点状态，如果未计算则返回None
        """
        if self._cached_strategy is None:
            return None
        
        return self._cached_strategy.node_state
    
    def select_hand(self, hand_label: str) -> Optional[HandStrategy]:
        """选择一个手牌查看详细策略。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            手牌策略
        """
        self._state.selected_hand = hand_label
        return self.get_hand_strategy(hand_label)
    
    def get_selected_hand(self) -> Optional[str]:
        """获取当前选中的手牌标签。
        
        Returns:
            手牌标签，如果未选中则返回None
        """
        return self._state.selected_hand
    
    # ========================================================================
    # 价值估计相关方法
    # ========================================================================
    
    def has_value_network(self) -> bool:
        """检查是否有价值网络可用。
        
        Returns:
            是否有价值网络
        """
        return self._value_calculator.is_available
    
    def get_hand_value(self, hand_label: str) -> Optional[HandValue]:
        """获取指定手牌的价值估计。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            手牌价值，如果未计算则返回None
        """
        if self._cached_value is None:
            return None
        
        return self._cached_value.hand_values.get(hand_label)
    
    def get_all_hand_values(self) -> Dict[str, HandValue]:
        """获取所有手牌的价值估计。
        
        Returns:
            手牌价值字典
        """
        if self._cached_value is None:
            return {}
        
        return self._cached_value.hand_values.copy()
    
    def get_value_result(self) -> Optional[ValueResult]:
        """获取当前的价值计算结果。
        
        Returns:
            价值计算结果
        """
        return self._cached_value
    
    # ========================================================================
    # 遗憾值估计相关方法
    # ========================================================================
    
    def has_regret_network(self) -> bool:
        """检查是否有遗憾网络可用。
        
        Returns:
            是否有遗憾网络
        """
        return self._regret_calculator.is_available
    
    def get_hand_regret(self, hand_label: str) -> Optional[HandRegret]:
        """获取指定手牌的遗憾值估计。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            手牌遗憾值，如果未计算则返回None
        """
        if self._cached_regret is None:
            return None
        
        return self._cached_regret.hand_regrets.get(hand_label)
    
    def get_all_hand_regrets(self) -> Dict[str, HandRegret]:
        """获取所有手牌的遗憾值估计。
        
        Returns:
            手牌遗憾值字典
        """
        if self._cached_regret is None:
            return {}
        
        return self._cached_regret.hand_regrets.copy()
    
    def get_regret_result(self) -> Optional[RegretResult]:
        """获取当前的遗憾值计算结果。
        
        Returns:
            遗憾值计算结果
        """
        return self._cached_regret
    
    # ========================================================================
    # 导出相关方法
    # ========================================================================
    
    def export_image(
        self, 
        path: Union[str, Path],
        title: Optional[str] = None
    ) -> str:
        """导出手牌矩阵图片。
        
        Args:
            path: 保存路径
            title: 图片标题（可选）
            
        Returns:
            保存的文件路径
            
        Raises:
            RuntimeError: 导出失败
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
        except ImportError:
            raise RuntimeError("需要安装matplotlib库来导出图片")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取颜色矩阵
        colors = self.get_matrix_colors()
        
        # 创建图像数据
        img_data = np.zeros((13, 13, 3), dtype=np.uint8)
        for i, row in enumerate(colors):
            for j, color in enumerate(row):
                img_data[i, j] = [color.r, color.g, color.b]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_data)
        
        # 添加标签
        for i, row in enumerate(HAND_LABELS_MATRIX):
            for j, label in enumerate(row):
                ax.text(j, i, label, ha='center', va='center', 
                       fontsize=8, color='white', fontweight='bold')
        
        # 设置标题
        if title:
            ax.set_title(title, fontsize=14)
        else:
            node = self.get_current_node()
            ax.set_title(f"策略矩阵 - {node.stage.value}", fontsize=14)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def export_json(self, path: Union[str, Path]) -> str:
        """导出策略数据为JSON文件。
        
        导出的JSON包含所有169种手牌的策略概率，以及模型的动作配置信息。
        每个手牌的策略包含与当前模型动作维度相同数量的动作概率。
        
        Args:
            path: 保存路径
            
        Returns:
            保存的文件路径
            
        需求引用:
        - 需求 5.3: 导出的JSON包含模型实际支持的所有动作
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取动作配置
        action_config = self.get_action_config()
        
        # 构建导出数据
        export_data = {
            "metadata": {
                "node_id": self._state.current_node_id,
                "board_cards": [
                    {"rank": c.rank, "suit": c.suit} 
                    for c in self._state.board_cards
                ],
                "model_loaded": self._state.is_model_loaded,
            },
            "strategies": {}
        }
        
        # 添加动作配置信息到元数据
        if action_config:
            export_data["metadata"]["action_config"] = {
                "action_names": action_config.action_names,
                "action_dim": action_config.action_dim,
            }
            if action_config.display_names:
                export_data["metadata"]["action_config"]["display_names"] = action_config.display_names
        
        # 添加模型元数据
        if self._model_loader.metadata:
            meta = self._model_loader.metadata
            export_data["metadata"]["model"] = {
                "checkpoint_path": meta.checkpoint_path,
                "episode_number": meta.episode_number,
                "timestamp": meta.timestamp,
            }
        
        # 添加所有手牌策略
        strategies = self.get_all_hand_strategies()
        for label, strategy in strategies.items():
            # 确保导出所有动作的概率（包括概率为0的动作）
            action_probabilities = strategy.action_probabilities.copy()
            
            # 如果有动作配置，确保所有动作都包含在导出数据中
            if action_config:
                for action_name in action_config.action_names:
                    if action_name not in action_probabilities:
                        action_probabilities[action_name] = 0.0
            
            export_data["strategies"][label] = {
                "action_probabilities": action_probabilities,
                "num_combinations": len(strategy.combinations),
                "is_pure_strategy": strategy.is_pure_strategy(),
                "dominant_action": strategy.get_dominant_action(),
            }
        
        # 写入文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(path)
    
    # ========================================================================
    # 事件回调设置
    # ========================================================================
    
    def set_on_model_loaded(
        self, 
        callback: Optional[Callable[[ModelMetadata], None]]
    ) -> None:
        """设置模型加载完成的回调。
        
        Args:
            callback: 回调函数，接收ModelMetadata参数
        """
        self._on_model_loaded = callback
    
    def set_on_node_changed(
        self, 
        callback: Optional[Callable[[GameTreeNode], None]]
    ) -> None:
        """设置节点变化的回调。
        
        Args:
            callback: 回调函数，接收GameTreeNode参数
        """
        self._on_node_changed = callback
    
    def set_on_strategy_updated(
        self, 
        callback: Optional[Callable[[StrategyResult], None]]
    ) -> None:
        """设置策略更新的回调。
        
        Args:
            callback: 回调函数，接收StrategyResult参数
        """
        self._on_strategy_updated = callback
    
    def set_on_board_changed(
        self, 
        callback: Optional[Callable[[List[Card]], None]]
    ) -> None:
        """设置公共牌变化的回调。
        
        Args:
            callback: 回调函数，接收Card列表参数
        """
        self._on_board_changed = callback
    
    def set_on_value_updated(
        self, 
        callback: Optional[Callable[[ValueResult], None]]
    ) -> None:
        """设置价值更新的回调。
        
        Args:
            callback: 回调函数，接收ValueResult参数
        """
        self._on_value_updated = callback
    
    def set_on_regret_updated(
        self, 
        callback: Optional[Callable[[RegretResult], None]]
    ) -> None:
        """设置遗憾值更新的回调。
        
        Args:
            callback: 回调函数，接收RegretResult参数
        """
        self._on_regret_updated = callback
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def get_hand_labels_matrix(self) -> List[List[str]]:
        """获取13x13手牌标签矩阵。
        
        Returns:
            手牌标签矩阵
        """
        return [row.copy() for row in HAND_LABELS_MATRIX]
    
    def get_all_hand_labels(self) -> List[str]:
        """获取所有169种手牌标签。
        
        Returns:
            手牌标签列表
        """
        return self._hand_range_calculator.get_all_hand_labels()
    
    def get_hand_combinations(self, hand_label: str) -> List:
        """获取手牌标签的所有花色组合。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            花色组合列表
        """
        return self._hand_range_calculator.get_all_hand_combinations(hand_label)
    
    def get_viewer_state(self) -> ViewerState:
        """获取查看器当前状态。
        
        Returns:
            查看器状态对象
        """
        return ViewerState(
            is_model_loaded=self._state.is_model_loaded,
            current_node_id=self._state.current_node_id,
            board_cards=self._state.board_cards.copy(),
            selected_hand=self._state.selected_hand
        )
    
    # ========================================================================
    # 私有方法
    # ========================================================================
    
    def _update_strategy(self) -> None:
        """更新当前节点的策略计算。"""
        if self._strategy_calculator is None:
            # 没有加载模型时，使用默认策略计算器
            self._strategy_calculator = StrategyCalculator()
        
        node = self._game_tree.get_current_node()
        
        # 计算策略
        self._cached_strategy = self._strategy_calculator.calculate_node_strategy(
            node=node,
            board_cards=self._state.board_cards if self._state.board_cards else None,
            player_id=max(0, node.player)  # 非玩家节点默认为玩家0
        )
        
        # 触发回调
        if self._on_strategy_updated:
            self._on_strategy_updated(self._cached_strategy)
        
        # 更新价值估计
        self._update_value()
    
    def _update_value(self) -> None:
        """更新当前节点的价值估计和遗憾值估计。
        
        注意：
        1. 价值/遗憾值估计始终从玩家0的视角计算，以保持热图的一致性
        2. 翻后阶段需要公共牌才能正确计算
        """
        node = self._game_tree.get_current_node()
        
        # 确定使用的公共牌
        # 优先使用用户选择的公共牌，否则使用节点的公共牌
        effective_board = self._state.board_cards if self._state.board_cards else node.board_cards
        
        # 更新价值估计（旧格式检查点）
        if self._value_calculator.is_available:
            self._cached_value = self._value_calculator.calculate_node_values(
                node=node,
                board_cards=effective_board if effective_board else None,
                player_id=0  # 始终从玩家0的视角
            )
            
            # 触发回调
            if self._on_value_updated:
                self._on_value_updated(self._cached_value)
        else:
            self._cached_value = None
        
        # 更新遗憾值估计（Deep CFR格式检查点）
        if self._regret_calculator.is_available:
            self._cached_regret = self._regret_calculator.calculate_node_regrets(
                node=node,
                board_cards=effective_board if effective_board else None,
                player_id=0  # 始终从玩家0的视角
            )
            
            # 触发回调
            if self._on_regret_updated:
                self._on_regret_updated(self._cached_regret)
        else:
            self._cached_regret = None
