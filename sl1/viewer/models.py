"""策略查看器核心数据模型。

本模块定义了策略查看器所需的核心数据类：
- GameTreeNode: 游戏树节点
- HandStrategy: 手牌策略信息
- NodeState: 节点状态信息
- ActionConfig: 动作空间配置
- BarSegment: 条状图段
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum

from models.core import Card, Action, GameStage


# 不同维度的默认动作映射
DEFAULT_ACTION_MAPPINGS: Dict[int, List[str]] = {
    4: ['FOLD', 'CHECK', 'CALL', 'RAISE'],
    5: ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG'],
    6: ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG', 'ALL_IN'],
}


@dataclass
class ActionConfig:
    """动作空间配置。
    
    封装模型的动作空间信息，包括动作名称列表和维度。
    支持从检查点数据创建或根据维度使用默认配置。
    
    Attributes:
        action_names: 动作名称列表，按索引顺序
        action_dim: 动作空间维度
        display_names: 显示用的动作名称映射（可选，用于合并CHECK/CALL等）
    """
    action_names: List[str]
    action_dim: int
    display_names: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """验证配置数据。"""
        # 验证动作名称列表长度与维度一致
        if len(self.action_names) != self.action_dim:
            raise ValueError(
                f"动作名称列表长度({len(self.action_names)})必须等于动作维度({self.action_dim})"
            )
        
        # 验证动作维度在支持范围内
        if self.action_dim < 1:
            raise ValueError(f"动作维度必须大于0，当前值: {self.action_dim}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_data: Dict[str, Any]) -> 'ActionConfig':
        """从检查点数据创建配置。
        
        如果检查点包含action_config元数据，则使用该配置；
        否则尝试从网络权重检测维度并使用默认映射。
        
        Args:
            checkpoint_data: 检查点数据字典
            
        Returns:
            ActionConfig实例
        """
        # 检查是否有action_config元数据
        if 'action_config' in checkpoint_data:
            config_data = checkpoint_data['action_config']
            action_names = config_data.get('action_names', []).copy()  # 复制列表以避免修改原始数据
            action_dim = config_data.get('action_dim', len(action_names))
            display_names = config_data.get('display_names', None)
            if display_names is not None:
                display_names = display_names.copy()  # 复制字典以避免修改原始数据
            return cls(
                action_names=action_names,
                action_dim=action_dim,
                display_names=display_names
            )
        
        # 尝试从网络权重检测维度
        action_dim = cls._detect_action_dim_from_weights(checkpoint_data)
        return cls.default_for_dim(action_dim)
    
    @classmethod
    def default_for_dim(cls, action_dim: int) -> 'ActionConfig':
        """根据维度创建默认配置。
        
        Args:
            action_dim: 动作空间维度
            
        Returns:
            ActionConfig实例
            
        Raises:
            ValueError: 如果维度不在支持的默认映射中
        """
        if action_dim in DEFAULT_ACTION_MAPPINGS:
            action_names = DEFAULT_ACTION_MAPPINGS[action_dim].copy()
        else:
            # 对于不在预定义映射中的维度，生成通用动作名称
            action_names = [f'ACTION_{i}' for i in range(action_dim)]
        
        return cls(
            action_names=action_names,
            action_dim=action_dim
        )
    
    @staticmethod
    def _detect_action_dim_from_weights(checkpoint_data: Dict[str, Any]) -> int:
        """从检查点的网络权重中检测动作维度。
        
        检查策略网络或遗憾网络的输出层维度。
        
        Args:
            checkpoint_data: 检查点数据字典
            
        Returns:
            检测到的动作维度，默认为6
        """
        # 尝试从策略网络检测
        if 'policy_net' in checkpoint_data:
            policy_state = checkpoint_data['policy_net']
            # 查找输出层权重（通常是最后一个带有'weight'的键）
            for key in reversed(list(policy_state.keys())):
                if 'weight' in key and 'output' in key.lower():
                    return policy_state[key].shape[0]
                # 也检查最后一层的形状
                if key.endswith('.weight'):
                    weight = policy_state[key]
                    # 输出层通常是较小的维度
                    if weight.shape[0] <= 10:
                        return weight.shape[0]
        
        # 尝试从遗憾网络检测
        if 'regret_net' in checkpoint_data:
            regret_state = checkpoint_data['regret_net']
            for key in reversed(list(regret_state.keys())):
                if 'weight' in key and 'output' in key.lower():
                    return regret_state[key].shape[0]
                if key.endswith('.weight'):
                    weight = regret_state[key]
                    if weight.shape[0] <= 10:
                        return weight.shape[0]
        
        # 检查是否有直接的action_dim字段
        if 'action_dim' in checkpoint_data:
            return checkpoint_data['action_dim']
        
        # 默认返回6维
        return 6
    
    def get_action_index(self, action_name: str) -> int:
        """获取动作名称对应的索引。
        
        Args:
            action_name: 动作名称
            
        Returns:
            动作索引
            
        Raises:
            ValueError: 如果动作名称不存在
        """
        try:
            return self.action_names.index(action_name)
        except ValueError:
            raise ValueError(f"未知的动作名称: {action_name}")
    
    def get_display_name(self, action_name: str) -> str:
        """获取动作的显示名称。
        
        Args:
            action_name: 动作名称
            
        Returns:
            显示名称（如果有映射则返回映射值，否则返回原名称）
        """
        if self.display_names and action_name in self.display_names:
            return self.display_names[action_name]
        return action_name


@dataclass
class BarSegment:
    """条状图的一个段。
    
    用于在手牌矩阵中显示策略分布的条状组合。
    
    Attributes:
        action: 动作名称
        probability: 概率值 (0.0 - 1.0)
        color: 颜色，格式为 (R, G, B) 元组
        width_ratio: 宽度比例 (0.0 - 1.0)
    """
    action: str
    probability: float
    color: Tuple[int, int, int]
    width_ratio: float
    
    def __post_init__(self):
        """验证段数据。"""
        # 验证概率范围
        if self.probability < 0.0 or self.probability > 1.0:
            raise ValueError(
                f"概率必须在[0.0, 1.0]范围内，当前值: {self.probability}"
            )
        
        # 验证宽度比例范围
        if self.width_ratio < 0.0 or self.width_ratio > 1.0:
            raise ValueError(
                f"宽度比例必须在[0.0, 1.0]范围内，当前值: {self.width_ratio}"
            )
        
        # 验证颜色格式
        if len(self.color) != 3:
            raise ValueError(f"颜色必须是(R, G, B)格式，当前值: {self.color}")
        for i, c in enumerate(self.color):
            if c < 0 or c > 255:
                raise ValueError(
                    f"颜色分量必须在[0, 255]范围内，分量{i}的值: {c}"
                )


class NodeType(Enum):
    """节点类型枚举。"""
    ROOT = "root"           # 根节点（游戏开始）
    CHANCE = "chance"       # 机会节点（发牌）
    PLAYER = "player"       # 玩家决策节点
    TERMINAL = "terminal"   # 终端节点（游戏结束）


@dataclass
class GameTreeNode:
    """游戏树节点，表示一个决策点。
    
    Attributes:
        node_id: 节点唯一标识符
        stage: 游戏阶段（preflop/flop/turn/river）
        player: 当前行动玩家（0或1，-1表示非玩家节点）
        action: 到达此节点的行动（可选）
        parent: 父节点（可选）
        children: 子节点列表
        pot: 当前底池大小
        stacks: 玩家筹码列表
        board_cards: 公共牌列表
        action_history: 行动历史
        node_type: 节点类型
    """
    node_id: str
    stage: GameStage
    player: int = -1  # -1表示非玩家节点（如根节点、机会节点）
    action: Optional[Action] = None
    parent: Optional['GameTreeNode'] = None
    children: List['GameTreeNode'] = field(default_factory=list)
    pot: int = 0
    stacks: List[int] = field(default_factory=lambda: [1000, 1000])
    board_cards: List[Card] = field(default_factory=list)
    action_history: List[Action] = field(default_factory=list)
    node_type: NodeType = NodeType.PLAYER
    
    def __post_init__(self):
        """验证节点数据。"""
        # 验证玩家编号
        if self.node_type == NodeType.PLAYER and self.player not in [0, 1]:
            raise ValueError(f"玩家节点的player必须是0或1，当前值: {self.player}")
        
        # 验证底池
        if self.pot < 0:
            raise ValueError(f"底池不能为负数，当前值: {self.pot}")
        
        # 验证筹码
        if len(self.stacks) != 2:
            raise ValueError(f"必须有2个玩家的筹码，当前数量: {len(self.stacks)}")
        for i, stack in enumerate(self.stacks):
            if stack < 0:
                raise ValueError(f"玩家{i}的筹码不能为负数，当前值: {stack}")
        
        # 验证公共牌数量
        if len(self.board_cards) > 5:
            raise ValueError(f"公共牌不能超过5张，当前数量: {len(self.board_cards)}")
    
    def __eq__(self, other) -> bool:
        """判断两个节点是否相等。"""
        if not isinstance(other, GameTreeNode):
            return False
        return self.node_id == other.node_id
    
    def __hash__(self) -> int:
        """节点哈希值。"""
        return hash(self.node_id)
    
    def is_terminal(self) -> bool:
        """判断是否为终端节点。"""
        return self.node_type == NodeType.TERMINAL
    
    def is_player_node(self) -> bool:
        """判断是否为玩家决策节点。"""
        return self.node_type == NodeType.PLAYER
    
    def get_depth(self) -> int:
        """获取节点深度（从根节点开始计算）。"""
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth
    
    def get_path_to_root(self) -> List['GameTreeNode']:
        """获取从当前节点到根节点的路径。
        
        Returns:
            节点列表，从根节点到当前节点的顺序
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path


@dataclass
class HandStrategy:
    """单个手牌的策略信息。
    
    Attributes:
        hand_label: 手牌标签（如"AKs"、"AKo"、"AA"）
        combinations: 具体花色组合列表
        action_probabilities: 各行动的概率 {action: probability}
        combination_strategies: 每个具体组合的策略（可选）
    """
    hand_label: str
    combinations: List[Tuple[Card, Card]]
    action_probabilities: Dict[str, float]
    combination_strategies: Optional[Dict[str, Dict[str, float]]] = None
    
    def __post_init__(self):
        """验证策略数据。"""
        # 验证手牌标签格式
        if not self._is_valid_hand_label(self.hand_label):
            raise ValueError(f"无效的手牌标签: {self.hand_label}")
        
        # 验证概率之和（允许浮点误差）
        total_prob = sum(self.action_probabilities.values())
        if abs(total_prob - 1.0) > 0.001:
            raise ValueError(
                f"行动概率之和必须为1.0，当前值: {total_prob}"
            )
        
        # 验证每个概率值在有效范围内
        for action, prob in self.action_probabilities.items():
            if prob < 0 or prob > 1:
                raise ValueError(
                    f"行动'{action}'的概率必须在[0, 1]范围内，当前值: {prob}"
                )
    
    def _is_valid_hand_label(self, label: str) -> bool:
        """验证手牌标签格式是否正确。
        
        有效格式：
        - 对子: "AA", "KK", ..., "22" (2个字符)
        - 同花: "AKs", "AQs", ..., "32s" (3个字符，以's'结尾)
        - 非同花: "AKo", "AQo", ..., "32o" (3个字符，以'o'结尾)
        """
        valid_ranks = set('AKQJT98765432')
        
        if len(label) == 2:
            # 对子格式
            return label[0] in valid_ranks and label[0] == label[1]
        elif len(label) == 3:
            # 同花或非同花格式
            return (label[0] in valid_ranks and 
                    label[1] in valid_ranks and 
                    label[0] != label[1] and
                    label[2] in ['s', 'o'])
        return False
    
    def get_dominant_action(self) -> str:
        """获取概率最高的行动。"""
        return max(self.action_probabilities.items(), key=lambda x: x[1])[0]
    
    def is_pure_strategy(self, threshold: float = 0.95) -> bool:
        """判断是否为纯策略（某个行动概率超过阈值）。"""
        return max(self.action_probabilities.values()) >= threshold


@dataclass
class NodeState:
    """节点的完整状态信息。
    
    Attributes:
        stage: 游戏阶段
        current_player: 当前玩家
        total_players: 总玩家数
        pot: 底池大小
        stacks: 玩家筹码
        board_cards: 公共牌
        action_history: 行动历史
        available_actions: 可用行动列表
    """
    stage: GameStage
    current_player: int
    total_players: int
    pot: int
    stacks: List[int]
    board_cards: List[Card]
    action_history: List[Action]
    available_actions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """验证状态数据。"""
        # 验证当前玩家
        if self.current_player < 0 or self.current_player >= self.total_players:
            raise ValueError(
                f"当前玩家编号无效: {self.current_player}，"
                f"有效范围: [0, {self.total_players - 1}]"
            )
        
        # 验证总玩家数
        if self.total_players != 2:
            raise ValueError(f"目前仅支持2人游戏，当前玩家数: {self.total_players}")
        
        # 验证底池
        if self.pot < 0:
            raise ValueError(f"底池不能为负数，当前值: {self.pot}")
        
        # 验证筹码列表长度
        if len(self.stacks) != self.total_players:
            raise ValueError(
                f"筹码列表长度({len(self.stacks)})必须等于玩家数({self.total_players})"
            )
        
        # 验证公共牌数量与游戏阶段的一致性
        # 注意：在策略查看器中，允许公共牌数量为0，因为用户需要手动选择公共牌
        num_board = len(self.board_cards)
        max_cards_by_stage = {
            GameStage.PREFLOP: 0,
            GameStage.FLOP: 3,
            GameStage.TURN: 4,
            GameStage.RIVER: 5,
        }
        max_cards = max_cards_by_stage.get(self.stage, 5)
        if num_board > max_cards:
            raise ValueError(
                f"在{self.stage.value}阶段，公共牌数量最多为{max_cards}张，"
                f"当前数量: {num_board}"
            )
    
    def get_stage_name(self) -> str:
        """获取游戏阶段的中文名称。"""
        stage_names = {
            GameStage.PREFLOP: "翻牌前",
            GameStage.FLOP: "翻牌",
            GameStage.TURN: "转牌",
            GameStage.RIVER: "河牌",
        }
        return stage_names.get(self.stage, str(self.stage))
    
    @classmethod
    def from_game_tree_node(cls, node: GameTreeNode) -> 'NodeState':
        """从GameTreeNode创建NodeState。
        
        Args:
            node: 游戏树节点
            
        Returns:
            节点状态对象
        """
        return cls(
            stage=node.stage,
            current_player=max(0, node.player),  # 非玩家节点默认为0
            total_players=2,
            pot=node.pot,
            stacks=node.stacks.copy(),
            board_cards=node.board_cards.copy(),
            action_history=node.action_history.copy(),
        )
