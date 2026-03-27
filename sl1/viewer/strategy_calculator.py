"""策略计算器模块。

本模块实现了策略计算功能：
- 集成现有的StrategyAnalyzer
- 计算节点策略
- 确保策略概率归一化
- 支持动态动作配置
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from models.core import Card, GameState, GameStage, Action, ActionType
from viewer.models import GameTreeNode, HandStrategy, NodeState, ActionConfig
from viewer.hand_range import HandRangeCalculator


@dataclass
class StrategyResult:
    """策略计算结果。
    
    Attributes:
        hand_strategies: 所有手牌的策略 {手牌标签: HandStrategy}
        available_actions: 可用行动列表
        node_state: 节点状态
    """
    hand_strategies: Dict[str, HandStrategy]
    available_actions: List[str]
    node_state: NodeState


class StrategyCalculator:
    """策略计算器 - 计算节点策略并确保概率归一化。
    
    提供以下功能：
    - 计算指定节点的所有手牌策略
    - 确保策略概率归一化
    - 支持公共牌过滤
    - 支持动态动作配置
    """
    
    def __init__(
        self, 
        strategy_analyzer: Optional[Any] = None,
        action_config: Optional[ActionConfig] = None,
        use_regret_network: bool = True
    ):
        """初始化策略计算器。
        
        Args:
            strategy_analyzer: 策略分析器实例（可选）
                如果提供，将使用它来获取模型策略
                如果不提供，将使用均匀分布作为默认策略
            action_config: 动作配置（可选）
                如果不提供，将使用默认的6维动作配置
            use_regret_network: 是否使用后悔值网络计算策略（默认True）
                - True: 使用后悔值网络的遗憾值通过Regret Matching计算策略（当前策略）
                - False: 使用策略网络的输出（平均策略）
        """
        self._analyzer = strategy_analyzer
        self._hand_range_calculator = HandRangeCalculator()
        self._use_regret_network = use_regret_network
        
        # 设置动作配置
        if action_config is not None:
            self._action_config = action_config
        else:
            # 默认使用6维动作配置
            self._action_config = ActionConfig.default_for_dim(6)
        
        # 构建显示用的动作列表（合并CHECK和CALL为CHECK/CALL）
        self._display_actions = self._build_display_actions()
    
    def _build_display_actions(self) -> List[str]:
        """构建显示用的动作列表。
        
        将CHECK和CALL合并为CHECK/CALL显示。
        
        Returns:
            显示用的动作名称列表
        """
        display_actions = []
        has_check = 'CHECK' in self._action_config.action_names
        has_call = 'CALL' in self._action_config.action_names
        
        for action in self._action_config.action_names:
            if action == 'CHECK' and has_call:
                # 如果同时有CHECK和CALL，合并为CHECK/CALL
                if 'CHECK/CALL' not in display_actions:
                    display_actions.append('CHECK/CALL')
            elif action == 'CALL' and has_check:
                # 已经在CHECK时添加了CHECK/CALL，跳过
                continue
            else:
                display_actions.append(action)
        
        return display_actions
    
    def set_action_config(self, config: ActionConfig) -> None:
        """设置动作配置。
        
        Args:
            config: 动作配置对象
        """
        self._action_config = config
        self._display_actions = self._build_display_actions()
    
    @property
    def available_actions(self) -> List[str]:
        """获取可用动作列表（显示用）。
        
        Returns:
            显示用的动作名称列表
        """
        return self._display_actions.copy()
    
    @property
    def action_config(self) -> ActionConfig:
        """获取当前动作配置。
        
        Returns:
            动作配置对象
        """
        return self._action_config
    
    def calculate_node_strategy(
        self,
        node: GameTreeNode,
        board_cards: Optional[List[Card]] = None,
        player_id: int = 0
    ) -> StrategyResult:
        """计算指定节点的所有手牌策略。
        
        Args:
            node: 游戏树节点
            board_cards: 公共牌列表（可选，如果不提供则使用节点的公共牌）
            player_id: 玩家ID
            
        Returns:
            策略计算结果
        """
        # 使用节点的公共牌或传入的公共牌
        effective_board = board_cards if board_cards is not None else node.board_cards
        
        # 获取所有手牌标签
        all_labels = self._hand_range_calculator.get_all_hand_labels()
        
        # 计算每个手牌的策略
        hand_strategies: Dict[str, HandStrategy] = {}
        
        for label in all_labels:
            strategy = self._calculate_hand_strategy(
                label, node, effective_board, player_id
            )
            hand_strategies[label] = strategy
        
        # 创建节点状态
        node_state = NodeState.from_game_tree_node(node)
        
        return StrategyResult(
            hand_strategies=hand_strategies,
            available_actions=self.available_actions,
            node_state=node_state
        )
    
    def _calculate_hand_strategy(
        self,
        hand_label: str,
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int
    ) -> HandStrategy:
        """计算单个手牌的策略。
        
        Args:
            hand_label: 手牌标签
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            
        Returns:
            手牌策略
        """
        # 获取该手牌的所有组合
        all_combinations = self._hand_range_calculator.get_all_hand_combinations(hand_label)
        
        # 过滤与公共牌冲突的组合
        valid_combinations = self._hand_range_calculator.filter_by_board(
            all_combinations, board_cards
        )
        
        # 计算策略
        if self._analyzer is not None and self._analyzer.is_model_loaded:
            # 使用模型计算策略
            action_probs = self._calculate_model_strategy(
                valid_combinations, node, board_cards, player_id
            )
        else:
            # 使用均匀分布作为默认策略
            action_probs = self._get_uniform_strategy()
        
        # 归一化策略概率
        action_probs = self.normalize_strategy(action_probs)
        
        # 计算每个组合的策略（如果有模型）
        combination_strategies = None
        if self._analyzer is not None and self._analyzer.is_model_loaded and valid_combinations:
            combination_strategies = self._calculate_combination_strategies(
                valid_combinations, node, board_cards, player_id
            )
        
        return HandStrategy(
            hand_label=hand_label,
            combinations=valid_combinations,
            action_probabilities=action_probs,
            combination_strategies=combination_strategies
        )
    
    def _calculate_model_strategy(
        self,
        combinations: List[Tuple[Card, Card]],
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int
    ) -> Dict[str, float]:
        """使用模型计算策略。
        
        对所有有效组合的策略取平均值。
        
        Args:
            combinations: 有效的手牌组合
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            
        Returns:
            行动概率字典
        """
        if not combinations:
            return self._get_uniform_strategy()
        
        # 累积所有组合的策略（使用显示用的动作列表）
        total_probs = {action: 0.0 for action in self._display_actions}
        
        for hand in combinations:
            # 创建游戏状态
            state = self._create_game_state(hand, node, board_cards, player_id)
            
            # 获取模型策略（使用配置的网络类型）
            probs = self._analyzer.analyze_state(
                state, player_id, use_regret_network=self._use_regret_network
            )
            
            for action, prob in probs.items():
                if action in total_probs:
                    total_probs[action] += prob
        
        # 取平均值
        num_combinations = len(combinations)
        avg_probs = {
            action: prob / num_combinations 
            for action, prob in total_probs.items()
        }
        
        return avg_probs
    
    def _calculate_combination_strategies(
        self,
        combinations: List[Tuple[Card, Card]],
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int
    ) -> Dict[str, Dict[str, float]]:
        """计算每个具体组合的策略。
        
        Args:
            combinations: 手牌组合列表
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            
        Returns:
            每个组合的策略 {组合字符串: {行动: 概率}}
        """
        result = {}
        
        for hand in combinations:
            # 创建组合的字符串表示
            combo_key = f"{hand[0]}{hand[1]}"
            
            # 创建游戏状态
            state = self._create_game_state(hand, node, board_cards, player_id)
            
            # 获取模型策略（使用配置的网络类型）
            probs = self._analyzer.analyze_state(
                state, player_id, use_regret_network=self._use_regret_network
            )
            
            # 归一化
            result[combo_key] = self.normalize_strategy(probs)
        
        return result
    
    def _create_game_state(
        self,
        hand: Tuple[Card, Card],
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int
    ) -> GameState:
        """创建游戏状态对象。
        
        根据节点的行动历史计算当前阶段的下注状态。
        
        Args:
            hand: 玩家手牌
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            
        Returns:
            游戏状态对象
        """
        # 为对手创建虚拟手牌
        opponent_hand = self._create_dummy_hand(hand, board_cards)
        
        if player_id == 0:
            player_hands = [hand, opponent_hand]
        else:
            player_hands = [opponent_hand, hand]
        
        # 计算当前阶段的下注
        current_bets = self._calculate_current_bets(node)
        
        return GameState(
            player_hands=player_hands,
            community_cards=board_cards,
            pot=node.pot,
            player_stacks=node.stacks.copy(),
            current_bets=current_bets,
            button_position=0,
            stage=node.stage,
            action_history=node.action_history.copy(),
            current_player=player_id
        )
    
    def _calculate_current_bets(self, node: GameTreeNode) -> List[int]:
        """计算当前阶段每个玩家的下注金额。
        
        Args:
            node: 游戏树节点
            
        Returns:
            当前下注列表 [玩家0下注, 玩家1下注]
        """
        from models.core import ActionType
        
        # 找到当前阶段开始的位置
        current_stage = node.stage
        stage_start_index = 0
        
        # 遍历父节点找到阶段变化点
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.stage != current_stage:
                stage_start_index = len(ancestor.action_history)
                break
            ancestor = ancestor.parent
        
        # 获取当前阶段的行动
        current_stage_actions = node.action_history[stage_start_index:]
        
        # 初始化下注
        if node.stage == GameStage.PREFLOP:
            bets = [5, 10]  # 小盲、大盲
        else:
            bets = [0, 0]
        
        # 根据当前阶段的行动更新下注
        if node.stage != GameStage.PREFLOP:
            current_actor = 1  # 翻牌后大盲位先行动
        else:
            current_actor = 0  # 翻牌前小盲位先行动
        
        for action in current_stage_actions:
            if action.action_type == ActionType.CALL:
                bets[current_actor] = max(bets)
            elif action.action_type in (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN):
                bets[current_actor] = max(bets) + action.amount
            current_actor = 1 - current_actor
        
        return bets
    
    def _create_dummy_hand(
        self,
        exclude_hand: Tuple[Card, Card],
        exclude_board: List[Card]
    ) -> Tuple[Card, Card]:
        """创建虚拟手牌（用于策略分析）。
        
        Args:
            exclude_hand: 要排除的手牌
            exclude_board: 要排除的公共牌
            
        Returns:
            虚拟手牌
        """
        excluded = set()
        excluded.add((exclude_hand[0].rank, exclude_hand[0].suit))
        excluded.add((exclude_hand[1].rank, exclude_hand[1].suit))
        for card in exclude_board:
            excluded.add((card.rank, card.suit))
        
        # 找两张不冲突的牌
        dummy_cards = []
        for rank in range(2, 15):
            for suit in ['h', 'd', 'c', 's']:
                if (rank, suit) not in excluded:
                    dummy_cards.append(Card(rank, suit))
                    if len(dummy_cards) == 2:
                        return (dummy_cards[0], dummy_cards[1])
        
        # 不应该到达这里
        raise RuntimeError("无法创建虚拟手牌")
    
    def _get_uniform_strategy(self) -> Dict[str, float]:
        """获取均匀分布策略。
        
        Returns:
            均匀分布的行动概率
        """
        num_actions = len(self._display_actions)
        prob = 1.0 / num_actions
        return {action: prob for action in self._display_actions}
    
    @staticmethod
    def normalize_strategy(strategy: Dict[str, float]) -> Dict[str, float]:
        """归一化策略概率，确保概率之和为1.0。
        
        Args:
            strategy: 原始策略概率字典
            
        Returns:
            归一化后的策略概率字典
        """
        if not strategy:
            return strategy
        
        total = sum(strategy.values())
        
        # 如果总和为0或非常小，返回均匀分布
        if total < 1e-10:
            num_actions = len(strategy)
            return {action: 1.0 / num_actions for action in strategy}
        
        # 归一化
        normalized = {action: prob / total for action, prob in strategy.items()}
        
        # 确保精确归一化（处理浮点误差）
        # 将误差加到最大概率的行动上
        current_total = sum(normalized.values())
        if abs(current_total - 1.0) > 1e-10:
            max_action = max(normalized.items(), key=lambda x: x[1])[0]
            normalized[max_action] += (1.0 - current_total)
        
        return normalized
    
    @staticmethod
    def is_normalized(strategy: Dict[str, float], tolerance: float = 0.001) -> bool:
        """检查策略是否已归一化。
        
        Args:
            strategy: 策略概率字典
            tolerance: 允许的误差范围
            
        Returns:
            是否已归一化
        """
        if not strategy:
            return True
        
        total = sum(strategy.values())
        return abs(total - 1.0) <= tolerance
