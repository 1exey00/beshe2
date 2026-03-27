"""手牌范围计算器模块。

本模块实现了手牌范围相关的计算功能：
- 手牌标签生成（如"AKs"、"AKo"、"AA"）
- 手牌组合枚举
- 公共牌冲突过滤
"""

from typing import List, Tuple, Dict, Optional
from models.core import Card


# 牌面等级映射（rank值到字符）
RANK_TO_CHAR = {
    14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
    9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'
}

# 字符到rank值的映射
CHAR_TO_RANK = {v: k for k, v in RANK_TO_CHAR.items()}

# 所有花色
SUITS = ['h', 'd', 'c', 's']

# 13x13手牌标签矩阵
# 行和列分别代表两张牌的rank（从A到2）
# 对角线: 对子 (AA, KK, QQ, ...)
# 上三角: 同花 (AKs, AQs, ...)
# 下三角: 非同花 (AKo, AQo, ...)
RANKS_ORDER = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

HAND_LABELS_MATRIX = [
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


class HandRangeCalculator:
    """手牌范围计算器 - 计算和管理手牌范围的策略。
    
    提供以下功能：
    - 获取手牌标签
    - 获取手牌的所有花色组合
    - 根据公共牌过滤手牌组合
    - 获取手牌在矩阵中的位置
    """
    
    def __init__(self):
        """初始化手牌范围计算器。"""
        # 构建标签到位置的映射
        self._label_to_position: Dict[str, Tuple[int, int]] = {}
        for row in range(13):
            for col in range(13):
                label = HAND_LABELS_MATRIX[row][col]
                self._label_to_position[label] = (row, col)
    
    def get_hand_label(self, card1: Card, card2: Card) -> str:
        """获取手牌的标准标签。
        
        规则：
        - 对子（两张牌rank相同）：返回"XX"格式（如"AA"）
        - 同花（两张牌suit相同且rank不同）：返回"XYs"格式（如"AKs"），高牌在前
        - 非同花（两张牌suit不同且rank不同）：返回"XYo"格式（如"AKo"），高牌在前
        
        Args:
            card1: 第一张牌
            card2: 第二张牌
            
        Returns:
            手牌标签字符串
        """
        rank1_char = RANK_TO_CHAR[card1.rank]
        rank2_char = RANK_TO_CHAR[card2.rank]
        
        # 确保高牌在前
        if card1.rank < card2.rank:
            rank1_char, rank2_char = rank2_char, rank1_char
        
        # 对子
        if card1.rank == card2.rank:
            return f"{rank1_char}{rank2_char}"
        
        # 同花
        if card1.suit == card2.suit:
            return f"{rank1_char}{rank2_char}s"
        
        # 非同花
        return f"{rank1_char}{rank2_char}o"
    
    def get_matrix_position(self, hand_label: str) -> Tuple[int, int]:
        """获取手牌标签在13x13矩阵中的位置。
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            (行, 列) 位置元组
            
        Raises:
            ValueError: 无效的手牌标签
        """
        if hand_label not in self._label_to_position:
            raise ValueError(f"无效的手牌标签: {hand_label}")
        return self._label_to_position[hand_label]
    
    def get_all_hand_combinations(self, hand_label: str) -> List[Tuple[Card, Card]]:
        """获取某个手牌标签的所有具体花色组合。
        
        组合数量：
        - 对子（如"AA"）：6种组合（C(4,2) = 6）
        - 同花（如"AKs"）：4种组合（4种花色）
        - 非同花（如"AKo"）：12种组合（4×3 = 12）
        
        Args:
            hand_label: 手牌标签
            
        Returns:
            所有具体花色组合的列表
            
        Raises:
            ValueError: 无效的手牌标签
        """
        if len(hand_label) == 2:
            # 对子
            rank_char = hand_label[0]
            if rank_char not in CHAR_TO_RANK:
                raise ValueError(f"无效的手牌标签: {hand_label}")
            rank = CHAR_TO_RANK[rank_char]
            return self._get_pair_combinations(rank)
        
        elif len(hand_label) == 3:
            rank1_char = hand_label[0]
            rank2_char = hand_label[1]
            suffix = hand_label[2]
            
            if rank1_char not in CHAR_TO_RANK or rank2_char not in CHAR_TO_RANK:
                raise ValueError(f"无效的手牌标签: {hand_label}")
            
            rank1 = CHAR_TO_RANK[rank1_char]
            rank2 = CHAR_TO_RANK[rank2_char]
            
            if suffix == 's':
                return self._get_suited_combinations(rank1, rank2)
            elif suffix == 'o':
                return self._get_offsuit_combinations(rank1, rank2)
            else:
                raise ValueError(f"无效的手牌标签: {hand_label}")
        
        else:
            raise ValueError(f"无效的手牌标签: {hand_label}")
    
    def filter_by_board(
        self, 
        combinations: List[Tuple[Card, Card]], 
        board_cards: List[Card]
    ) -> List[Tuple[Card, Card]]:
        """过滤与公共牌冲突的手牌组合。
        
        如果手牌中的任何一张牌与公共牌中的任何一张牌相同，
        则该组合被过滤掉。
        
        Args:
            combinations: 手牌组合列表
            board_cards: 公共牌列表
            
        Returns:
            过滤后的手牌组合列表
        """
        if not board_cards:
            return combinations
        
        # 构建公共牌集合（使用(rank, suit)元组）
        board_set = {(card.rank, card.suit) for card in board_cards}
        
        filtered = []
        for card1, card2 in combinations:
            # 检查手牌是否与公共牌冲突
            if ((card1.rank, card1.suit) not in board_set and 
                (card2.rank, card2.suit) not in board_set):
                filtered.append((card1, card2))
        
        return filtered
    
    def get_all_hand_labels(self) -> List[str]:
        """获取所有169种手牌标签。
        
        Returns:
            所有手牌标签的列表（按矩阵顺序）
        """
        labels = []
        for row in HAND_LABELS_MATRIX:
            labels.extend(row)
        return labels
    
    def is_pair(self, hand_label: str) -> bool:
        """判断手牌标签是否为对子。"""
        return len(hand_label) == 2
    
    def is_suited(self, hand_label: str) -> bool:
        """判断手牌标签是否为同花。"""
        return len(hand_label) == 3 and hand_label[2] == 's'
    
    def is_offsuit(self, hand_label: str) -> bool:
        """判断手牌标签是否为非同花。"""
        return len(hand_label) == 3 and hand_label[2] == 'o'
    
    def _get_pair_combinations(self, rank: int) -> List[Tuple[Card, Card]]:
        """获取对子的所有花色组合（6种）。"""
        combinations = []
        for i, suit1 in enumerate(SUITS):
            for suit2 in SUITS[i + 1:]:
                combinations.append((Card(rank, suit1), Card(rank, suit2)))
        return combinations
    
    def _get_suited_combinations(self, rank1: int, rank2: int) -> List[Tuple[Card, Card]]:
        """获取同花的所有花色组合（4种）。"""
        combinations = []
        for suit in SUITS:
            combinations.append((Card(rank1, suit), Card(rank2, suit)))
        return combinations
    
    def _get_offsuit_combinations(self, rank1: int, rank2: int) -> List[Tuple[Card, Card]]:
        """获取非同花的所有花色组合（12种）。"""
        combinations = []
        for suit1 in SUITS:
            for suit2 in SUITS:
                if suit1 != suit2:
                    combinations.append((Card(rank1, suit1), Card(rank2, suit2)))
        return combinations
