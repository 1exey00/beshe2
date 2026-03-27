"""CFR (Counterfactual Regret Minimization) 训练器实现。

CFR是一种用于求解博弈论中纳什均衡的算法，特别适用于不完全信息博弈如德州扑克。

支持两种信息集计算模式：
1. 原始模式：使用具体手牌和公共牌
2. 抽象模式：使用预计算的卡牌抽象桶ID，大幅减少信息集数量
"""

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy

from models.core import GameState, Action, ActionType, GameStage, Card

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from abstraction.card_abstraction import CardAbstraction


@dataclass(frozen=True)
class InfoSet:
    """信息集（Information Set）表示玩家在某一时刻所知道的所有信息。
    
    在德州扑克中，信息集包括：
    - 玩家的手牌（或抽象后的桶ID）
    - 公共牌（或抽象后的桶ID）
    - 下注历史
    - 当前游戏阶段
    
    支持两种模式：
    1. 原始模式：hand_key和community_key存储具体牌
    2. 抽象模式：bucket_id存储抽象后的桶ID
    
    Attributes:
        hand_key: 手牌的规范化表示（排序后的牌），抽象模式下为空元组
        community_key: 公共牌的规范化表示，抽象模式下为空元组
        stage: 游戏阶段
        action_history_key: 行动历史的规范化表示
        pot_ratio: 底池与初始筹码的比例（离散化）
        bucket_id: 抽象后的桶ID（仅在抽象模式下使用，原始模式下为-1）
    """
    hand_key: Tuple[Tuple[int, str], ...]
    community_key: Tuple[Tuple[int, str], ...]
    stage: str
    action_history_key: Tuple[str, ...]
    pot_ratio: int  # 离散化的底池比例
    bucket_id: int = -1  # 抽象后的桶ID（-1表示未使用抽象）
    
    def __hash__(self):
        if self.bucket_id >= 0:
            # 抽象模式：使用桶ID作为主要标识
            return hash((self.bucket_id, self.stage, 
                         self.action_history_key, self.pot_ratio))
        else:
            # 原始模式：使用具体手牌和公共牌
            return hash((self.hand_key, self.community_key, self.stage, 
                         self.action_history_key, self.pot_ratio))
    
    def __eq__(self, other):
        if not isinstance(other, InfoSet):
            return False
        if self.bucket_id >= 0 and other.bucket_id >= 0:
            # 抽象模式：比较桶ID
            return (self.bucket_id == other.bucket_id and
                    self.stage == other.stage and
                    self.action_history_key == other.action_history_key and
                    self.pot_ratio == other.pot_ratio)
        else:
            # 原始模式：比较具体牌
            return (self.hand_key == other.hand_key and 
                    self.community_key == other.community_key and
                    self.stage == other.stage and
                    self.action_history_key == other.action_history_key and
                    self.pot_ratio == other.pot_ratio)
    
    @property
    def is_abstracted(self) -> bool:
        """检查是否使用了抽象。"""
        return self.bucket_id >= 0


class CFRTrainer:
    """CFR（反事实遗憾最小化）训练器。
    
    实现了标准的CFR算法，用于训练德州扑克AI。
    
    支持两种信息集计算模式：
    1. 原始模式：使用具体手牌和公共牌
    2. 抽象模式：使用预计算的卡牌抽象桶ID，大幅减少信息集数量
    
    Attributes:
        regret_sum: 累积遗憾值表，键为信息集，值为各行动的遗憾值数组
        strategy_sum: 累积策略表，用于计算平均策略
        num_actions: 行动空间大小
        iterations: 已执行的迭代次数
        card_abstraction: 可选的卡牌抽象对象
        use_abstraction: 是否使用卡牌抽象
    """
    
    # 标准行动类型
    ACTION_TYPES = [
        ActionType.FOLD,
        ActionType.CHECK,
        ActionType.CALL,
        ActionType.RAISE_SMALL,
        ActionType.RAISE_BIG,
        ActionType.ALL_IN
    ]
    
    def __init__(self, num_actions: int = 4, initial_stack: int = 1000,
                 card_abstraction: Optional['CardAbstraction'] = None):
        """初始化CFR训练器。
        
        Args:
            num_actions: 行动空间大小（默认4：弃牌、过牌、跟注、加注）
            initial_stack: 初始筹码，用于计算底池比例
            card_abstraction: 可选的卡牌抽象对象，用于抽象信息集
        """
        self.num_actions = num_actions
        self.initial_stack = initial_stack
        self.card_abstraction = card_abstraction
        self.use_abstraction = (card_abstraction is not None and 
                                 card_abstraction.is_loaded())
        
        # 遗憾值表：信息集 -> 各行动的累积遗憾值
        self.regret_sum: Dict[InfoSet, np.ndarray] = {}
        
        # 策略累积表：信息集 -> 各行动的累积策略权重
        self.strategy_sum: Dict[InfoSet, np.ndarray] = {}
        
        # 迭代计数
        self.iterations = 0
    
    def get_info_set(self, state: GameState, player_id: int) -> InfoSet:
        """从游戏状态创建信息集。
        
        将游戏状态抽象为信息集，相似的状态会被归为同一信息集。
        
        支持两种模式：
        1. 原始模式：使用具体手牌和公共牌
        2. 抽象模式：使用预计算的卡牌抽象桶ID
        
        Args:
            state: 当前游戏状态
            player_id: 玩家ID（0或1）
            
        Returns:
            对应的信息集
        """
        # 游戏阶段
        stage = state.stage.value
        
        # 行动历史抽象（只保留行动类型，不保留具体金额）
        action_history = tuple(a.action_type.value for a in state.action_history)
        
        # 底池比例离散化（分为10个档位）
        pot_ratio = min(9, state.pot * 10 // (self.initial_stack * 2))
        
        # 检查是否使用卡牌抽象
        if self.use_abstraction and self.card_abstraction is not None:
            # 抽象模式：使用桶ID
            hand = state.player_hands[player_id]
            hole_cards = (hand[0], hand[1])
            community_cards = list(state.community_cards)
            
            # 获取桶ID
            bucket_id = self.card_abstraction.get_bucket_id(hole_cards, community_cards)
            
            return InfoSet(
                hand_key=(),  # 抽象模式下不使用具体手牌
                community_key=(),  # 抽象模式下不使用具体公共牌
                stage=stage,
                action_history_key=action_history,
                pot_ratio=pot_ratio,
                bucket_id=bucket_id
            )
        else:
            # 原始模式：使用具体手牌和公共牌
            hand = state.player_hands[player_id]
            hand_cards = sorted([(c.rank, c.suit) for c in hand], reverse=True)
            hand_key = tuple(hand_cards)
            
            # 获取公共牌并规范化
            community_cards = sorted([(c.rank, c.suit) for c in state.community_cards], reverse=True)
            community_key = tuple(community_cards)
            
            return InfoSet(
                hand_key=hand_key,
                community_key=community_key,
                stage=stage,
                action_history_key=action_history,
                pot_ratio=pot_ratio,
                bucket_id=-1  # 原始模式下不使用桶ID
            )
    
    def set_card_abstraction(self, card_abstraction: 'CardAbstraction') -> None:
        """设置卡牌抽象对象。
        
        Args:
            card_abstraction: 卡牌抽象对象
        """
        self.card_abstraction = card_abstraction
        self.use_abstraction = (card_abstraction is not None and 
                                 card_abstraction.is_loaded())
    
    def enable_abstraction(self, enable: bool = True) -> None:
        """启用或禁用卡牌抽象。
        
        Args:
            enable: 是否启用抽象
        """
        if enable and self.card_abstraction is None:
            raise ValueError("无法启用抽象：卡牌抽象对象未设置")
        if enable and not self.card_abstraction.is_loaded():
            raise ValueError("无法启用抽象：卡牌抽象未加载")
        self.use_abstraction = enable
    
    def get_abstraction_stats(self) -> Dict[str, int]:
        """获取抽象相关的统计信息。
        
        Returns:
            包含以下信息的字典：
            - total_info_sets: 总信息集数量
            - abstracted_info_sets: 使用抽象的信息集数量
            - raw_info_sets: 使用原始表示的信息集数量
        """
        total = len(self.regret_sum)
        abstracted = sum(1 for info_set in self.regret_sum.keys() if info_set.is_abstracted)
        raw = total - abstracted
        
        return {
            'total_info_sets': total,
            'abstracted_info_sets': abstracted,
            'raw_info_sets': raw,
            'use_abstraction': self.use_abstraction
        }
    
    def get_strategy(self, info_set: InfoSet) -> np.ndarray:
        """获取当前信息集的策略（使用Regret Matching）。
        
        Regret Matching算法：
        - 如果某个行动的累积遗憾值为正，则该行动的概率与遗憾值成正比
        - 如果所有行动的遗憾值都为非正，则使用均匀分布
        
        Args:
            info_set: 信息集
            
        Returns:
            行动概率分布（numpy数组）
        """
        # 获取累积遗憾值，如果不存在则初始化为0
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = np.zeros(self.num_actions)
        
        regrets = self.regret_sum[info_set]
        
        # Regret Matching: 只考虑正遗憾值
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            # 按正遗憾值比例分配概率
            strategy = positive_regrets / regret_sum
        else:
            # 所有遗憾值都为非正，使用均匀分布
            strategy = np.ones(self.num_actions) / self.num_actions
        
        return strategy

    
    def compute_regrets(self, state: GameState, player_id: int, 
                        reach_probs: Tuple[float, float],
                        action_utilities: Dict[int, float]) -> Dict[InfoSet, np.ndarray]:
        """计算反事实遗憾值。
        
        反事实遗憾值 = 选择该行动的期望收益 - 当前策略的期望收益
        
        Args:
            state: 当前游戏状态
            player_id: 当前玩家ID
            reach_probs: 两个玩家到达当前状态的概率
            action_utilities: 各行动的效用值字典
            
        Returns:
            信息集到遗憾值数组的映射
        """
        info_set = self.get_info_set(state, player_id)
        strategy = self.get_strategy(info_set)  #初始时为平均概率
        
        # 计算当前策略的期望效用
        expected_utility = 0.0
        for action_idx, utility in action_utilities.items():
            expected_utility += strategy[action_idx] * utility
        
        # 计算各行动的遗憾值
        regrets = np.zeros(self.num_actions)
        for action_idx, utility in action_utilities.items():
            # 反事实遗憾 = 该行动的效用 - 当前策略的期望效用
            regrets[action_idx] = utility - expected_utility
        
        # 用对手的到达概率加权（反事实权重）
        opponent_id = 1 - player_id
        counterfactual_weight = reach_probs[opponent_id]
        weighted_regrets = regrets * counterfactual_weight
        
        return {info_set: weighted_regrets}
    
    def update_strategy(self, regrets: Dict[InfoSet, np.ndarray]) -> None:
        """基于遗憾值更新策略。
        
        将新的遗憾值累加到遗憾值表中，并更新策略累积表。
        
        Args:
            regrets: 信息集到遗憾值数组的映射
        """
        for info_set, regret_values in regrets.items():
            # 累加遗憾值
            if info_set not in self.regret_sum:
                self.regret_sum[info_set] = np.zeros(self.num_actions)
            self.regret_sum[info_set] += regret_values
            
            # 获取当前策略并累加到策略表
            strategy = self.get_strategy(info_set)
            if info_set not in self.strategy_sum:
                self.strategy_sum[info_set] = np.zeros(self.num_actions)
            self.strategy_sum[info_set] += strategy
        
        self.iterations += 1
    
    def get_average_strategy(self, info_set: Optional[InfoSet] = None) -> Dict[InfoSet, np.ndarray]:
        """获取累积平均策略。
        
        平均策略是CFR算法收敛到的纳什均衡策略。
        
        Args:
            info_set: 如果指定，只返回该信息集的平均策略；否则返回所有信息集的平均策略
            
        Returns:
            信息集到平均策略的映射
        """
        if info_set is not None:
            # 返回单个信息集的平均策略
            if info_set not in self.strategy_sum:
                # 如果没有累积策略，返回均匀分布
                return {info_set: np.ones(self.num_actions) / self.num_actions}
            
            strategy_sum = self.strategy_sum[info_set]
            total = np.sum(strategy_sum)
            
            if total > 0:
                return {info_set: strategy_sum / total}
            else:
                return {info_set: np.ones(self.num_actions) / self.num_actions}
        
        # 返回所有信息集的平均策略
        average_strategies = {}
        for info_set, strategy_sum in self.strategy_sum.items():
            total = np.sum(strategy_sum)
            if total > 0:
                average_strategies[info_set] = strategy_sum / total
            else:
                average_strategies[info_set] = np.ones(self.num_actions) / self.num_actions
        
        return average_strategies

    
    def get_action_from_strategy(self, strategy: np.ndarray, 
                                  legal_actions: List[Action]) -> Action:
        """根据策略概率分布选择行动。
        
        Args:
            strategy: 行动概率分布
            legal_actions: 合法行动列表
            
        Returns:
            选择的行动
        """
        # 创建合法行动的概率分布
        legal_probs = []
        legal_indices = []
        
        for action in legal_actions:
            action_idx = self._action_to_index(action)
            legal_probs.append(strategy[action_idx])
            legal_indices.append(action_idx)
        
        # 归一化概率
        total_prob = sum(legal_probs)
        if total_prob > 0:
            legal_probs = [p / total_prob for p in legal_probs]
        else:
            # 如果所有合法行动概率为0，使用均匀分布
            legal_probs = [1.0 / len(legal_actions)] * len(legal_actions)
        
        # 按概率选择行动
        choice = np.random.choice(len(legal_actions), p=legal_probs)
        return legal_actions[choice]
    
    def _action_to_index(self, action: Action) -> int:
        """将行动转换为索引。
        
        Args:
            action: 行动对象
            
        Returns:
            行动索引
        """
        action_type_to_idx = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.RAISE_SMALL: 3,
            ActionType.RAISE_BIG: 4,
            ActionType.RAISE: 3,  # 向后兼容
            ActionType.ALL_IN: 5
        }
        return action_type_to_idx.get(action.action_type, 0)
    
    def _index_to_action_type(self, index: int) -> ActionType:
        """将索引转换为行动类型。
        
        Args:
            index: 行动索引
            
        Returns:
            行动类型
        """
        idx_to_action_type = {
            0: ActionType.FOLD,
            1: ActionType.CHECK,
            2: ActionType.CALL,
            3: ActionType.RAISE_SMALL,
            4: ActionType.RAISE_BIG,
            5: ActionType.ALL_IN
        }
        return idx_to_action_type.get(index, ActionType.FOLD)
    
    def reset(self) -> None:
        """重置训练器状态。"""
        self.regret_sum.clear()
        self.strategy_sum.clear()
        self.iterations = 0
    
    def get_num_info_sets(self) -> int:
        """获取已访问的信息集数量。
        
        Returns:
            信息集数量
        """
        return len(self.regret_sum)
    
    def get_cfr_guided_target(self, state: GameState, player_id: int, 
                               legal_action_indices: List[int]) -> np.ndarray:
        """获取CFR引导的目标策略分布，用于神经网络训练。
        
        该方法返回基于CFR平均策略的目标分布，用于指导神经网络学习。
        如果信息集没有足够的历史数据，则返回均匀分布。
        
        Args:
            state: 当前游戏状态
            player_id: 玩家ID
            legal_action_indices: 合法行动的索引列表
            
        Returns:
            目标策略分布（numpy数组，长度为num_actions）
        """
        info_set = self.get_info_set(state, player_id)
        
        # 获取平均策略
        avg_strategy = self.get_average_strategy(info_set)
        strategy = avg_strategy.get(info_set, np.ones(self.num_actions) / self.num_actions)
        
        # 确保只有合法行动有非零概率
        masked_strategy = np.zeros(self.num_actions)
        for idx in legal_action_indices:
            masked_strategy[idx] = strategy[idx]
        
        # 归一化
        total = np.sum(masked_strategy)
        if total > 0:
            masked_strategy = masked_strategy / total
        else:
            # 如果所有合法行动概率为0，使用均匀分布
            for idx in legal_action_indices:
                masked_strategy[idx] = 1.0 / len(legal_action_indices)
        
        return masked_strategy


    def compute_and_update_regrets(self, state: GameState, player_id: int,
                                    action_taken: int, reward: float,
                                    legal_action_indices: List[int]) -> None:
        """计算并更新反事实遗憾值。

        基于实际执行的行动和获得的奖励，计算反事实遗憾值。
        这是一个简化版本的CFR更新，适用于与神经网络结合使用。

        关键改进：正确估计FOLD行动的遗憾值，防止FOLD策略消失。

        Args:
            state: 当前游戏状态
            player_id: 玩家ID
            action_taken: 实际执行的行动索引
            reward: 获得的奖励
            legal_action_indices: 合法行动的索引列表
        """
        info_set = self.get_info_set(state, player_id)

        # 获取当前策略、初始时为平均策略
        strategy = self.get_strategy(info_set)

        # 计算各行动的遗憾值
        # 核心思想：如果当前行动导致负奖励，那么FOLD可能是更好的选择
        regrets = np.zeros(self.num_actions)

        for idx in legal_action_indices:
            if idx == action_taken:
                # 实际执行的行动：遗憾值 = 0
                regrets[idx] = 0
            elif idx == 0:  # FOLD
                # FOLD的效用估计：
                # - 如果当前奖励为负（输钱），FOLD可能是更好的选择
                # - 如果当前奖励为正（赢钱），FOLD会损失这些收益
                # FOLD的效用 = 0（不赢不输）
                # 遗憾值 = FOLD效用 - 当前行动效用 = 0 - reward = -reward
                # 如果reward < 0，则-reward > 0，表示应该弃牌
                regrets[idx] = -reward
            else:
                # 其他行动：使用启发式估计
                # CHECK/CALL: 假设效用与当前行动相似
                # RAISE: 如果赢了，加注可能赢更多；如果输了，加注可能输更多
                if idx == 3:  # RAISE
                    # 加注的遗憾值：如果赢了，可能应该加注更多
                    regrets[idx] = max(0, reward * 0.2)
                else:  # CHECK/CALL
                    regrets[idx] = 0

        # 更新遗憾值表
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = np.zeros(self.num_actions)
        self.regret_sum[info_set] += regrets

        # 更新策略累积表
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = np.zeros(self.num_actions)
        self.strategy_sum[info_set] += strategy

        self.iterations += 1
