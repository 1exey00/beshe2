"""遗憾值计算器模块。

本模块实现了遗憾网络的遗憾值估计计算功能：
- 计算所有手牌的遗憾值估计
- 支持公共牌过滤
- 提供遗憾值热图数据

需求引用:
- 需求 9.2: 显示13x13手牌范围矩阵的遗憾值估计热图
- 需求 9.4: 显示该手牌的具体遗憾值估计数值
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch

from models.core import Card, GameState, GameStage, Action, ActionType
from models.networks import RegretNetwork
from environment.state_encoder import StateEncoder
from viewer.models import GameTreeNode, NodeState
from viewer.hand_range import HandRangeCalculator


@dataclass
class HandRegret:
    """单个手牌的遗憾值估计信息。
    
    Attributes:
        hand_label: 手牌标签（如"AKs"）
        combinations: 具体花色组合列表
        average_regrets: 所有组合的平均遗憾值（每个动作）
        average_total_regret: 所有动作遗憾值的平均绝对值
        combination_regrets: 每个具体组合的遗憾值（可选）
        is_blocked: 是否被公共牌阻挡
    """
    hand_label: str
    combinations: List[Tuple[Card, Card]]
    average_regrets: Dict[str, float]  # {action_name: regret_value}
    average_total_regret: float
    combination_regrets: Optional[Dict[str, Dict[str, float]]] = None
    is_blocked: bool = False


@dataclass
class RegretResult:
    """遗憾值计算结果。
    
    Attributes:
        hand_regrets: 所有手牌的遗憾值 {手牌标签: HandRegret}
        min_regret: 最小遗憾值
        max_regret: 最大遗憾值
        node_state: 节点状态
        valid_actions: 当前节点的有效动作列表
    """
    hand_regrets: Dict[str, HandRegret]
    min_regret: float
    max_regret: float
    node_state: NodeState
    valid_actions: List[str] = None  # 有效动作列表
    
    def __post_init__(self):
        if self.valid_actions is None:
            # 默认 6 维动作
            self.valid_actions = ['弃牌', '过牌', '跟注', '小加注', '大加注', '全下']


class RegretCalculator:
    """遗憾值计算器 - 计算手牌的遗憾值估计。
    
    提供以下功能：
    - 计算指定节点的所有手牌遗憾值估计
    - 支持公共牌过滤
    - 提供遗憾值热图数据
    - 根据游戏状态过滤有效动作
    - 支持动态动作配置
    """
    
    # 默认动作名称映射（6 维输出）
    DEFAULT_ACTION_NAMES = ['弃牌', '过牌', '跟注', '小加注', '大加注', '全下']
    # 默认动作索引映射
    DEFAULT_ACTION_INDEX = {'弃牌': 0, '过牌': 1, '跟注': 2, '小加注': 3, '大加注': 4, '全下': 5}
    
    # 动作名称翻译映射（英文到中文）
    ACTION_NAME_TRANSLATION = {
        'FOLD': '弃牌',
        'CHECK': '过牌',
        'CALL': '跟注',
        'RAISE': '加注',
        'RAISE_SMALL': '小加注',
        'RAISE_BIG': '大加注',
        'ALL_IN': '全下',
    }
    
    def __init__(self, regret_network: Optional[RegretNetwork] = None):
        """初始化遗憾值计算器。
        
        Args:
            regret_network: 遗憾网络实例（可选）
        """
        self._regret_network = regret_network
        self._state_encoder = StateEncoder()
        self._hand_range_calculator = HandRangeCalculator()
        self._action_names = self.DEFAULT_ACTION_NAMES.copy()
        self._action_index = self.DEFAULT_ACTION_INDEX.copy()
        self._action_dim = 6
    
    def set_action_config(self, action_names: List[str], action_dim: int) -> None:
        """设置动作配置。
        
        Args:
            action_names: 动作名称列表（英文）
            action_dim: 动作维度
        """
        self._action_dim = action_dim
        # 将英文动作名称翻译为中文
        self._action_names = [
            self.ACTION_NAME_TRANSLATION.get(name, name) 
            for name in action_names
        ]
        # 重建动作索引映射
        self._action_index = {name: i for i, name in enumerate(self._action_names)}
    
    @property
    def action_names(self) -> List[str]:
        """获取当前动作名称列表。"""
        return self._action_names
    
    @property
    def action_dim(self) -> int:
        """获取当前动作维度。"""
        return self._action_dim
    
    def set_regret_network(self, regret_network: Optional[RegretNetwork]) -> None:
        """设置遗憾网络。
        
        Args:
            regret_network: 遗憾网络实例
        """
        self._regret_network = regret_network
    
    @property
    def is_available(self) -> bool:
        """检查遗憾网络是否可用。"""
        return self._regret_network is not None
    
    def calculate_node_regrets(
        self,
        node: GameTreeNode,
        board_cards: Optional[List[Card]] = None,
        player_id: int = 0
    ) -> RegretResult:
        """计算指定节点的所有手牌遗憾值估计。
        
        注意：翻后阶段（FLOP/TURN/RIVER）需要公共牌才能正确计算遗憾值。
        如果没有提供公共牌，遗憾值估计可能不准确。
        
        Args:
            node: 游戏树节点
            board_cards: 公共牌列表（可选，如果不提供则使用节点的公共牌）
            player_id: 玩家ID
            
        Returns:
            遗憾值计算结果
        """
        # 使用节点的公共牌或传入的公共牌
        effective_board = board_cards if board_cards is not None else node.board_cards
        
        # 确定当前节点的有效动作
        valid_actions = self._get_valid_actions(node, effective_board, player_id)
        
        # 获取所有手牌标签
        all_labels = self._hand_range_calculator.get_all_hand_labels()
        
        # 计算每个手牌的遗憾值
        hand_regrets: Dict[str, HandRegret] = {}
        all_total_regrets = []
        
        for label in all_labels:
            hand_regret = self._calculate_hand_regret(
                label, node, effective_board, player_id, valid_actions
            )
            hand_regrets[label] = hand_regret
            if not hand_regret.is_blocked:
                all_total_regrets.append(hand_regret.average_total_regret)
        
        # 计算最小和最大遗憾值
        min_regret = min(all_total_regrets) if all_total_regrets else 0.0
        max_regret = max(all_total_regrets) if all_total_regrets else 0.0
        
        # 创建节点状态
        node_state = NodeState.from_game_tree_node(node)
        
        return RegretResult(
            hand_regrets=hand_regrets,
            min_regret=min_regret,
            max_regret=max_regret,
            node_state=node_state,
            valid_actions=valid_actions
        )
    
    def _get_valid_actions(
        self,
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int
    ) -> List[str]:
        """根据游戏状态确定有效动作。
        
        德州扑克规则：
        - 如果可以过牌（check），则不应该有弃牌（fold）选项
        - 如果需要跟注（对手有下注），则有弃牌、跟注、加注选项
        - 加注分为小加注（半底池）和大加注（全底池）
        - 全下（ALL_IN）总是可用的
        
        Args:
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            
        Returns:
            有效动作名称列表
        """
        # 计算当前下注状态
        current_bets = self._calculate_current_bets(node)
        current_player_bet = current_bets[player_id]
        opponent_bet = current_bets[1 - player_id]
        
        valid_actions = []
        
        # 如果下注相等，可以过牌，不能弃牌
        if current_player_bet == opponent_bet:
            valid_actions.append('过牌')
        else:
            # 需要跟注，可以弃牌
            valid_actions.append('弃牌')
            valid_actions.append('跟注')
        
        # 加注总是可以的（如果有筹码）- 两种加注大小
        valid_actions.append('小加注')
        valid_actions.append('大加注')
        
        # 如果动作配置包含全下，也添加
        if '全下' in self._action_names:
            valid_actions.append('全下')
        
        return valid_actions
    
    def _calculate_current_bets(self, node: GameTreeNode) -> List[int]:
        """计算当前阶段每个玩家的下注金额。
        
        Args:
            node: 游戏树节点
            
        Returns:
            [玩家0下注, 玩家1下注]
        """
        # 找到当前阶段开始的位置
        current_stage = node.stage
        stage_start_index = 0
        
        # 从节点向上回溯，找到阶段不同的祖先节点
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.stage != current_stage:
                stage_start_index = len(ancestor.action_history)
                break
            ancestor = ancestor.parent
        
        # 获取当前阶段的行动
        current_stage_actions = node.action_history[stage_start_index:]
        
        # 初始下注状态
        # 翻牌前：小盲5，大盲10（假设标准盲注）
        # 翻牌后：都是0
        if node.stage == GameStage.PREFLOP:
            # 从节点的筹码和底池推断盲注
            # 假设初始筹码是1000，小盲5，大盲10
            bets = [5, 10]  # 默认盲注
        else:
            bets = [0, 0]
        
        # 根据当前阶段的行动历史更新下注
        if node.stage != GameStage.PREFLOP:
            current_actor = 1  # 翻牌后大盲位先行动
        else:
            current_actor = 0  # 翻牌前小盲位先行动
        
        for action in current_stage_actions:
            if action.action_type == ActionType.CALL:
                bets[current_actor] = max(bets)
            elif action.action_type in (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG):
                bets[current_actor] = max(bets) + action.amount
            # CHECK 和 FOLD 不改变下注
            current_actor = 1 - current_actor
        
        return bets
    
    def _calculate_hand_regret(
        self,
        hand_label: str,
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int,
        valid_actions: Optional[List[str]] = None
    ) -> HandRegret:
        """计算单个手牌的遗憾值估计。
        
        Args:
            hand_label: 手牌标签
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            valid_actions: 有效动作列表（可选）
            
        Returns:
            手牌遗憾值
        """
        if valid_actions is None:
            valid_actions = self._action_names
        
        # 获取该手牌的所有组合
        all_combinations = self._hand_range_calculator.get_all_hand_combinations(hand_label)
        
        # 过滤与公共牌冲突的组合
        valid_combinations = self._hand_range_calculator.filter_by_board(
            all_combinations, board_cards
        )
        
        # 如果没有有效组合，标记为被阻挡
        if not valid_combinations:
            return HandRegret(
                hand_label=hand_label,
                combinations=[],
                average_regrets={name: 0.0 for name in valid_actions},
                average_total_regret=0.0,
                is_blocked=True
            )
        
        # 计算遗憾值
        if self._regret_network is not None:
            avg_regrets, total_regret, combination_regrets = self._calculate_model_regret(
                valid_combinations, node, board_cards, player_id, valid_actions
            )
        else:
            # 没有遗憾网络时返回0
            avg_regrets = {name: 0.0 for name in valid_actions}
            total_regret = 0.0
            combination_regrets = None
        
        return HandRegret(
            hand_label=hand_label,
            combinations=valid_combinations,
            average_regrets=avg_regrets,
            average_total_regret=total_regret,
            combination_regrets=combination_regrets,
            is_blocked=False
        )
    
    def _calculate_model_regret(
        self,
        combinations: List[Tuple[Card, Card]],
        node: GameTreeNode,
        board_cards: List[Card],
        player_id: int,
        valid_actions: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], float, Dict[str, Dict[str, float]]]:
        """使用遗憾网络计算遗憾值。
        
        Args:
            combinations: 有效的手牌组合
            node: 游戏树节点
            board_cards: 公共牌
            player_id: 玩家ID
            valid_actions: 有效动作列表（可选）
            
        Returns:
            (平均遗憾值字典, 总遗憾值, 每个组合的遗憾值字典)
        """
        if valid_actions is None:
            valid_actions = self._action_names
        
        if not combinations:
            return {name: 0.0 for name in valid_actions}, 0.0, {}
        
        # 获取有效动作的索引
        valid_indices = [self._action_index[name] for name in valid_actions if name in self._action_index]
        
        combination_regrets = {}
        total_regrets = np.zeros(self._action_dim)  # 使用动态维度
        
        for hand in combinations:
            # 创建游戏状态
            state = self._create_game_state(hand, node, board_cards, player_id)
            
            # 编码状态
            state_encoding = self._state_encoder.encode(state, player_id)
            state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
            
            # 获取遗憾值估计
            with torch.no_grad():
                regrets = self._regret_network(state_tensor).squeeze(0).numpy()
            
            # 记录组合遗憾值（只记录有效动作）
            combo_key = f"{hand[0]}{hand[1]}"
            combination_regrets[combo_key] = {
                name: float(regrets[self._action_index[name]]) 
                for name in valid_actions if name in self._action_index
            }
            total_regrets += regrets
        
        # 计算平均遗憾值（只返回有效动作）
        avg_regrets = total_regrets / len(combinations)
        avg_regrets_dict = {
            name: float(avg_regrets[self._action_index[name]]) 
            for name in valid_actions if name in self._action_index
        }
        
        # 计算总遗憾值（只计算有效动作的正遗憾值）
        valid_avg_regrets = np.array([avg_regrets[i] for i in valid_indices])
        positive_regrets = np.maximum(valid_avg_regrets, 0)
        total_positive_regret = float(np.sum(positive_regrets))
        
        return avg_regrets_dict, total_positive_regret, combination_regrets
    
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
        """创建虚拟手牌（用于遗憾值估计）。
        
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
    
    def get_regret_for_hand(
        self,
        hand_label: str,
        node: GameTreeNode,
        board_cards: Optional[List[Card]] = None,
        player_id: int = 0
    ) -> Optional[HandRegret]:
        """获取单个手牌的遗憾值估计。
        
        Args:
            hand_label: 手牌标签
            node: 游戏树节点
            board_cards: 公共牌列表
            player_id: 玩家ID
            
        Returns:
            手牌遗憾值，如果遗憾网络不可用则返回None
        """
        if not self.is_available:
            return None
        
        effective_board = board_cards if board_cards is not None else node.board_cards
        valid_actions = self._get_valid_actions(node, effective_board, player_id)
        return self._calculate_hand_regret(hand_label, node, effective_board, player_id, valid_actions)
