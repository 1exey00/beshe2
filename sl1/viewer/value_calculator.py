"""价值计算器模块。

本模块实现了价值网络的价值估计计算功能：
- 计算所有手牌的价值估计
- 支持公共牌过滤
- 提供价值热图数据

需求引用:
- 需求 9.2: 显示13x13手牌范围矩阵的价值估计热图
- 需求 9.4: 显示该手牌的具体价值估计数值
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch

from models.core import Card, GameState, GameStage, Action
from models.networks import ValueNetwork
from environment.state_encoder import StateEncoder
from viewer.models import GameTreeNode, NodeState
from viewer.hand_range import HandRangeCalculator


@dataclass
class HandValue:
    """单个手牌的价值估计信息。
    
    Attributes:
        hand_label: 手牌标签（如"AKs"）
        combinations: 具体花色组合列表
        average_value: 所有组合的平均价值
        combination_values: 每个具体组合的价值（可选）
        is_blocked: 是否被公共牌阻挡
    """
    hand_label: str
    combinations: List[Tuple[Card, Card]]
    average_value: float
    combination_values: Optional[Dict[str, float]] = None
    is_blocked: bool = False


@dataclass
class ValueResult:
    """价值计算结果。
    
    Attributes:
        hand_values: 所有手牌的价值 {手牌标签: HandValue}
        min_value: 最小价值
        max_value: 最大价值
        node_state: 节点状态
    """
    hand_values: Dict[str, HandValue]
    min_value: float
    max_value: float
    node_state: NodeState


class ValueCalculator:
    """价值计算器 - 计算手牌的价值估计。
    
    提供以下功能：
    - 计算指定节点的所有手牌价值估计
    - 支持公共牌过滤
    - 提供价值热图数据
    """
    
    def __init__(self, value_network: Optional[ValueNetwork] = None):
        """初始化价值计算器。
        
        Args:
            value_network: 价值网络实例（可选）
        """
        self._value_network = value_network
        self._state_encoder = StateEncoder()
        self._hand_range_calculator = HandRangeCalculator()
    
    def set_value_network(self, value_network: Optional[ValueNetwork]) -> None:
        """设置价值网络。
        
        Args:
            value_network: 价值网络实例
        """
        self._value_network = value_network
    
    @property
    def is_available(self) -> bool:
        """检查价值网络是否可用。"""
        return self._value_network is not None
    
    def calculate_node_values(
        self,
        node: GameTreeNode,
        board_cards: Optional[List[Card]] = None,
        player_id: int = 0
    ) -> ValueResult:
        """计算指定节点的所有手牌价值估计。
        
        注意：翻后阶段（FLOP/TURN/RIVER）需要公共牌才能正确计算价值。
        如果没有提供公共牌，价值估计可能不准确。
        
        Args:
            node: 游戏树节点
            board_cards: 公共牌列表（可选，如果不提供则使用节点的公共牌）
            player_id: 玩家ID
            
        Returns:
            价值计算结果
        """
        # 使用节点的公共牌或传入的公共牌
        effective_board = board_cards if board_cards is not None else node.board_cards
        
        # 检查翻后阶段是否有足够的公共牌
        # 翻后阶段需要公共牌才能正确计算价值
        is_postflop = node.stage in [GameStage.FLOP, GameStage.TURN, GameStage.RIVER]
        has_board = effective_board and len(effective_board) > 0
        
        if is_postflop and not has_board:
            # 翻后阶段没有公共牌，价值估计可能不准确
            # 但仍然计算，让用户知道需要选择公共牌
            pass
        
        # 获取所有手牌标签
        all_labels = self._hand_range_calculator.get_all_hand_labels()
        
        # 计算每个手牌的价值
        hand_values: Dict[str, HandValue] = {}
        all_values = []
        
        for label in all_labels:
            hand_value = self._calculate_hand_value(
                label, node, effective_board, player_id
            )
            hand_values[label] = hand_value
            if not hand_value.is_blocked:
                all_values.append(hand_value.average_value)
        
        # 计算最小和最大价值
        min_value = min(all_values) if all_values else 0.0
        max_value = max(all_values) if all_values else 0.0
        
        # 创建节点状态
        node_state = NodeState.from_game_tree_node(node)
        
        return ValueResult(
            hand_values=hand_values,
            min_value=min_value,
            max_value=max_value,
            node_state=node_state
        )
    
    def _calculate_hand_value(
        self,
        hand_label: str,
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int
    ) -> HandValue:
        """计算单个手牌的价值估计。
        
        Args:
            hand_label: 手牌标签
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            
        Returns:
            手牌价值
        """
        # 获取该手牌的所有组合
        all_combinations = self._hand_range_calculator.get_all_hand_combinations(hand_label)
        
        # 过滤与公共牌冲突的组合
        valid_combinations = self._hand_range_calculator.filter_by_board(
            all_combinations, board_cards
        )
        
        # 如果没有有效组合，标记为被阻挡
        if not valid_combinations:
            return HandValue(
                hand_label=hand_label,
                combinations=[],
                average_value=0.0,
                is_blocked=True
            )
        
        # 计算价值
        if self._value_network is not None:
            avg_value, combination_values = self._calculate_model_value(
                valid_combinations, node, board_cards, player_id
            )
        else:
            # 没有价值网络时返回0
            avg_value = 0.0
            combination_values = None
        
        return HandValue(
            hand_label=hand_label,
            combinations=valid_combinations,
            average_value=avg_value,
            combination_values=combination_values,
            is_blocked=False
        )
    
    def _calculate_model_value(
        self,
        combinations: List[Tuple[Card, Card]],
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int
    ) -> Tuple[float, Dict[str, float]]:
        """使用价值网络计算价值。
        
        Args:
            combinations: 有效的手牌组合
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            
        Returns:
            (平均价值, 每个组合的价值字典)
        """
        if not combinations:
            return 0.0, {}
        
        combination_values = {}
        total_value = 0.0
        
        for hand in combinations:
            # 创建游戏状态
            state = self._create_game_state(hand, node, board_cards, player_id)
            
            # 编码状态
            state_encoding = self._state_encoder.encode(state, player_id)
            state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
            
            # 获取价值估计
            with torch.no_grad():
                value = self._value_network(state_tensor).item()
            
            # 记录组合价值
            combo_key = f"{hand[0]}{hand[1]}"
            combination_values[combo_key] = value
            total_value += value
        
        # 计算平均价值
        avg_value = total_value / len(combinations)
        
        return avg_value, combination_values
    
    def _create_game_state(
        self,
        hand: Tuple[Card, Card],
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int
    ) -> GameState:
        """创建游戏状态对象。
        
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
        
        return GameState(
            player_hands=player_hands,
            community_cards=board_cards,
            pot=node.pot,
            player_stacks=node.stacks.copy(),
            current_bets=[0, 0],
            button_position=0,
            stage=node.stage,
            action_history=node.action_history.copy(),
            current_player=player_id
        )
    
    def _create_dummy_hand(
        self,
        exclude_hand: Tuple[Card, Card],
        exclude_board: List[Card]
    ) -> Tuple[Card, Card]:
        """创建虚拟手牌（用于价值估计）。
        
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
    
    def get_value_for_hand(
        self,
        hand_label: str,
        node: GameTreeNode,
        board_cards: Optional[List[Card]] = None,
        player_id: int = 0
    ) -> Optional[HandValue]:
        """获取单个手牌的价值估计。
        
        Args:
            hand_label: 手牌标签
            node: 游戏树节点
            board_cards: 公共牌列表
            player_id: 玩家ID
            
        Returns:
            手牌价值，如果价值网络不可用则返回None
        """
        if not self.is_available:
            return None
        
        effective_board = board_cards if board_cards is not None else node.board_cards
        return self._calculate_hand_value(hand_label, node, effective_board, player_id)
