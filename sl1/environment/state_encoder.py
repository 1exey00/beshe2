"""状态编码器 - 将游戏状态转换为神经网络输入。

本模块实现了三种编码方案：
1. 完整编码（370维）：用于翻牌后，保留花色信息
2. 抽象编码（175维）：用于翻牌前，将1326种手牌抽象为169种等价类
3. 桶抽象编码：使用预计算的卡牌抽象，将手牌+公共牌编码为桶ID的one-hot编码
"""

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

from models.core import GameState, Card, GameStage

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from abstraction.card_abstraction import CardAbstraction


class StateEncoder:
    """状态编码器 - 将扑克游戏状态编码为固定维度的特征向量。
    
    完整编码方案（370维）：
    - 玩家手牌: 104维（2张牌 × 52维）
    - 公共牌: 260维（5张牌 × 52维）
    - 筹码信息: 4维（玩家筹码、对手筹码、底池、当前下注）
    - 位置信息: 2维（是否按钮位、是否当前玩家）
    
    抽象编码方案（175维）- 用于翻牌前：
    - 手牌等价类: 169维（13×13矩阵的one-hot编码）
    - 筹码信息: 4维
    - 位置信息: 2维
    
    桶抽象编码方案（可变维度）- 使用预计算的卡牌抽象：
    - 桶ID: N维 one-hot编码（N为该阶段的桶数量）
    - 筹码信息: 4维
    - 位置信息: 2维
    
    169种等价类说明：
    - 对角线（13种）：对子 AA, KK, ..., 22
    - 上三角（78种）：同花牌 AKs, AQs, ...
    - 下三角（78种）：非同花牌 AKo, AQo, ...
    """
    
    def __init__(self, use_abstraction: bool = True, 
                 card_abstraction: Optional['CardAbstraction'] = None):
        """初始化状态编码器。
        
        Args:
            use_abstraction: 是否在翻牌前使用手牌抽象（默认True）
            card_abstraction: 可选的卡牌抽象对象，用于桶抽象编码
        """
        self.use_abstraction = use_abstraction
        self.card_abstraction = card_abstraction
        
        # 完整编码维度
        self.card_dim = 52
        self.hand_dim = 104  # 2张牌 × 52
        self.community_dim = 260  # 5张牌 × 52
        self.chip_dim = 4
        self.position_dim = 2
        self.full_encoding_dim = 370
        
        # 抽象编码维度
        self.hand_abstraction_dim = 169  # 13 × 13 等价类
        self.abstract_encoding_dim = 175  # 169 + 4 + 2
        
        # 默认使用完整编码维度（保持向后兼容）
        self.encoding_dim = self.full_encoding_dim
        
        # 桶抽象编码的最大桶数（用于确定编码维度）
        self._max_buckets = 5000  # 默认最大桶数
        
        # 构建等价类索引映射
        self._build_hand_abstraction_map()
    
    def _build_hand_abstraction_map(self) -> None:
        """构建手牌等价类索引映射。
        
        169种等价类的索引规则：
        - 索引 = row * 13 + col
        - row = 较大牌的rank索引（A=12, K=11, ..., 2=0）
        - col = 较小牌的rank索引
        - 对角线（row == col）：对子
        - 上三角（row > col）：同花牌
        - 下三角（row < col）：非同花牌
        """
        # rank名称到索引的映射（2=0, 3=1, ..., A=12）
        self.rank_to_index = {r: r - 2 for r in range(2, 15)}
        
        # 索引到手牌标签的映射（用于调试）
        rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.index_to_label = {}
        for i in range(169):
            row = i // 13
            col = i % 13
            if row == col:
                # 对子
                self.index_to_label[i] = f"{rank_names[row]}{rank_names[col]}"
            elif row > col:
                # 同花（较大牌在前）
                self.index_to_label[i] = f"{rank_names[row]}{rank_names[col]}s"
            else:
                # 非同花（较大牌在前）
                self.index_to_label[i] = f"{rank_names[col]}{rank_names[row]}o"
    
    def get_hand_abstraction_index(self, card1: Card, card2: Card) -> int:
        """获取手牌的等价类索引。
        
        Args:
            card1: 第一张牌
            card2: 第二张牌
            
        Returns:
            等价类索引（0-168）
        """
        rank1_idx = self.rank_to_index[card1.rank]
        rank2_idx = self.rank_to_index[card2.rank]
        suited = card1.suit == card2.suit
        
        # 确保较大的rank在前
        high_idx = max(rank1_idx, rank2_idx)
        low_idx = min(rank1_idx, rank2_idx)
        
        if high_idx == low_idx:
            # 对子：对角线
            return high_idx * 13 + low_idx
        elif suited:
            # 同花：上三角（row > col）
            return high_idx * 13 + low_idx
        else:
            # 非同花：下三角（row < col）
            return low_idx * 13 + high_idx
    
    def encode(self, state: GameState, player_id: int) -> np.ndarray:
        """将游戏状态编码为固定维度的特征向量。
        
        根据游戏阶段自动选择编码方案：
        - 翻牌前且启用抽象：使用175维抽象编码
        - 其他情况：使用370维完整编码
        
        Args:
            state: 游戏状态
            player_id: 玩家视角（0或1）
            
        Returns:
            编码后的特征向量
        """
        if player_id not in [0, 1]:
            raise ValueError(f"玩家ID必须是0或1，收到 {player_id}")
        
        # 翻牌前使用抽象编码
        if self.use_abstraction and state.stage == GameStage.PREFLOP:
            return self._encode_abstract(state, player_id)
        else:
            return self._encode_full(state, player_id)
    
    def _encode_abstract(self, state: GameState, player_id: int) -> np.ndarray:
        """翻牌前抽象编码。
        
        使用169维手牌等价类编码替代104维的完整手牌编码，
        但保持总维度为370以兼容现有网络。
        
        编码结构（370维）：
        - 手牌等价类: 169维 one-hot（替代原来的104维手牌编码）
        - 填充: 195维零（保持总维度不变）
        - 筹码信息: 4维
        - 位置信息: 2维
        
        Args:
            state: 游戏状态
            player_id: 玩家视角
            
        Returns:
            370维特征向量（与完整编码维度相同）
        """
        encoding = np.zeros(self.full_encoding_dim, dtype=np.float32)
        
        # 编码手牌等价类（169维 one-hot，放在前169维）
        hand = state.player_hands[player_id]
        abstraction_idx = self.get_hand_abstraction_index(hand[0], hand[1])
        encoding[abstraction_idx] = 1.0
        
        # 跳过手牌和公共牌区域（104 + 260 = 364维），直接到筹码信息
        offset = self.hand_dim + self.community_dim  # 364
        
        # 编码筹码信息（4维）
        max_stack = 2000
        encoding[offset] = state.player_stacks[player_id] / max_stack
        encoding[offset + 1] = state.player_stacks[1 - player_id] / max_stack
        encoding[offset + 2] = state.pot / max_stack
        encoding[offset + 3] = state.current_bets[player_id] / max_stack
        offset += self.chip_dim
        
        # 编码位置信息（2维）
        encoding[offset] = 1.0 if state.button_position == player_id else 0.0
        encoding[offset + 1] = 1.0 if state.current_player == player_id else 0.0
        
        return encoding
    
    def _encode_full(self, state: GameState, player_id: int) -> np.ndarray:
        """完整编码（370维）。
        
        Args:
            state: 游戏状态
            player_id: 玩家视角
            
        Returns:
            370维特征向量
        """
        encoding = np.zeros(self.full_encoding_dim, dtype=np.float32)
        offset = 0
        
        # 编码玩家手牌（104维）
        hand = state.player_hands[player_id]
        hand_encoding = self.encode_cards([hand[0], hand[1]])
        encoding[offset:offset + self.hand_dim] = hand_encoding
        offset += self.hand_dim
        
        # 编码公共牌（260维）
        community_encoding = self.encode_cards(state.community_cards, max_cards=5)
        encoding[offset:offset + self.community_dim] = community_encoding
        offset += self.community_dim
        
        # 编码筹码信息（4维）
        max_stack = 2000
        encoding[offset] = state.player_stacks[player_id] / max_stack
        encoding[offset + 1] = state.player_stacks[1 - player_id] / max_stack
        encoding[offset + 2] = state.pot / max_stack
        encoding[offset + 3] = state.current_bets[player_id] / max_stack
        offset += self.chip_dim
        
        # 编码位置信息（2维）
        encoding[offset] = 1.0 if state.button_position == player_id else 0.0
        encoding[offset + 1] = 1.0 if state.current_player == player_id else 0.0
        
        return encoding
    
    def encode_cards(self, cards: List[Card], max_cards: int = None) -> np.ndarray:
        """使用one-hot编码对牌列表进行编码。
        
        每张牌编码为52维one-hot向量：
        - 位置 = (rank - 2) * 4 + suit_index
        - rank范围: 2-14
        - suit_index: h=0, d=1, c=2, s=3
        
        Args:
            cards: 要编码的牌列表
            max_cards: 最大牌数（不足时用0填充）
            
        Returns:
            形状为 (max_cards * 52,) 的numpy数组
        """
        if max_cards is None:
            max_cards = len(cards)
        
        encoding = np.zeros(max_cards * self.card_dim, dtype=np.float32)
        
        suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}
        
        for i, card in enumerate(cards):
            if i >= max_cards:
                break
            
            # 计算在52维空间中的位置
            rank_index = card.rank - 2  # 0-12
            suit_index = suit_map[card.suit]  # 0-3
            card_position = rank_index * 4 + suit_index  # 0-51
            
            offset = i * self.card_dim
            encoding[offset + card_position] = 1.0
        
        return encoding
    
    def get_hand_label(self, card1: Card, card2: Card) -> str:
        """获取手牌的标签（如 'AKs', 'QQ', '72o'）。
        
        Args:
            card1: 第一张牌
            card2: 第二张牌
            
        Returns:
            手牌标签字符串
        """
        idx = self.get_hand_abstraction_index(card1, card2)
        return self.index_to_label[idx]
    
    def get_encoding_dim(self, stage: GameStage = None) -> int:
        """获取指定阶段的编码维度。
        
        Args:
            stage: 游戏阶段（如果为None则返回最大维度）
            
        Returns:
            编码维度
        """
        if self.use_abstraction and stage == GameStage.PREFLOP:
            return self.abstract_encoding_dim
        return self.full_encoding_dim
    
    def set_card_abstraction(self, card_abstraction: 'CardAbstraction') -> None:
        """设置卡牌抽象对象。
        
        Args:
            card_abstraction: 卡牌抽象对象
        """
        self.card_abstraction = card_abstraction
        
        # 更新最大桶数
        if card_abstraction is not None and card_abstraction.result is not None:
            config = card_abstraction.result.config
            self._max_buckets = max(
                config.preflop_buckets,
                config.flop_buckets,
                config.turn_buckets,
                config.river_buckets
            )
    
    def encode_with_abstraction(self, state: GameState, player_id: int) -> np.ndarray:
        """使用卡牌抽象将游戏状态编码为特征向量。
        
        将手牌+公共牌编码替换为桶ID的one-hot编码。
        编码维度固定为370维以保持与现有网络的兼容性。
        
        编码结构（370维）：
        - 桶ID: 最多364维 one-hot编码（使用前N维，N为桶数量）
        - 筹码信息: 4维
        - 位置信息: 2维
        
        Args:
            state: 游戏状态
            player_id: 玩家视角（0或1）
            
        Returns:
            编码后的特征向量（370维）
            
        Raises:
            ValueError: 如果卡牌抽象未设置或未加载
        """
        if player_id not in [0, 1]:
            raise ValueError(f"玩家ID必须是0或1，收到 {player_id}")
        
        if self.card_abstraction is None:
            raise ValueError("卡牌抽象未设置，请先调用set_card_abstraction()")
        
        if not self.card_abstraction.is_loaded():
            raise ValueError("卡牌抽象未加载，请先调用card_abstraction.load()或generate_abstraction()")
        
        encoding = np.zeros(self.full_encoding_dim, dtype=np.float32)
        
        # 获取手牌和公共牌
        hand = state.player_hands[player_id]
        hole_cards = (hand[0], hand[1])
        community_cards = list(state.community_cards)
        
        # 获取桶ID
        bucket_id = self.card_abstraction.get_bucket_id(hole_cards, community_cards)
        
        # 获取当前阶段的桶数量
        num_community = len(community_cards)
        stage_name = self.card_abstraction.get_stage_from_community_cards(num_community)
        bucket_count = self._get_bucket_count_for_stage(stage_name)
        
        # 桶ID的one-hot编码（放在前364维内）
        # 确保桶ID在有效范围内
        max_bucket_dim = self.hand_dim + self.community_dim  # 364
        if bucket_id < max_bucket_dim:
            encoding[bucket_id] = 1.0
        else:
            # 如果桶ID超出范围，使用模运算
            encoding[bucket_id % max_bucket_dim] = 1.0
        
        # 编码筹码信息（4维）
        offset = self.hand_dim + self.community_dim  # 364
        max_stack = 2000
        encoding[offset] = state.player_stacks[player_id] / max_stack
        encoding[offset + 1] = state.player_stacks[1 - player_id] / max_stack
        encoding[offset + 2] = state.pot / max_stack
        encoding[offset + 3] = state.current_bets[player_id] / max_stack
        offset += self.chip_dim
        
        # 编码位置信息（2维）
        encoding[offset] = 1.0 if state.button_position == player_id else 0.0
        encoding[offset + 1] = 1.0 if state.current_player == player_id else 0.0
        
        return encoding
    
    def _get_bucket_count_for_stage(self, stage_name: str) -> int:
        """获取指定阶段的桶数量。
        
        Args:
            stage_name: 阶段名称（'preflop', 'flop', 'turn', 'river'）
            
        Returns:
            桶数量
        """
        if self.card_abstraction is None or self.card_abstraction.result is None:
            return self._max_buckets
        
        config = self.card_abstraction.result.config
        stage_to_buckets = {
            'preflop': config.preflop_buckets,
            'flop': config.flop_buckets,
            'turn': config.turn_buckets,
            'river': config.river_buckets,
        }
        return stage_to_buckets.get(stage_name, self._max_buckets)
    
    def has_card_abstraction(self) -> bool:
        """检查是否设置了卡牌抽象。
        
        Returns:
            如果设置了卡牌抽象且已加载，返回True
        """
        return (self.card_abstraction is not None and 
                self.card_abstraction.is_loaded())
