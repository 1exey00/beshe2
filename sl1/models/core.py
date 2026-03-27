"""Core data classes for Texas Hold'em AI training system."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class CFRVariantType(Enum):
    """CFR变体类型枚举。
    
    定义了四种CFR变体：
    - STANDARD: 标准CFR算法
    - CFR_PLUS: CFR+，使用正遗憾值截断
    - LCFR: 线性CFR，使用线性迭代加权
    - DCFR: 折扣CFR，使用折扣因子
    """
    STANDARD = "standard"
    CFR_PLUS = "cfr_plus"
    LCFR = "lcfr"
    DCFR = "dcfr"


class ActionType(Enum):
    """Types of actions available in poker.
    
    动作类型：
    - FOLD: 弃牌
    - CHECK: 过牌
    - CALL: 跟注
    - RAISE: 加注（保留用于向后兼容）
    - RAISE_SMALL: 小加注（半底池）
    - RAISE_BIG: 大加注（全底池）
    - ALL_IN: 全下（当筹码不足以进行标准加注时）
    """
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"  # 保留用于向后兼容
    RAISE_SMALL = "raise_small"  # 半底池加注
    RAISE_BIG = "raise_big"  # 全底池加注
    ALL_IN = "all_in"  # 全下


class GameStage(Enum):
    """Stages of a poker hand."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class HandRank(Enum):
    """Poker hand rankings."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


@dataclass
class Card:
    """Represents a playing card.
    
    Attributes:
        rank: Card rank (2-14, where 11=J, 12=Q, 13=K, 14=A)
        suit: Card suit ('h'=hearts, 'd'=diamonds, 'c'=clubs, 's'=spades)
    """
    rank: int  # 2-14 (2-10, J=11, Q=12, K=13, A=14)
    suit: str  # 'h', 'd', 'c', 's'
    
    def __post_init__(self):
        """Validate card values."""
        if not 2 <= self.rank <= 14:
            raise ValueError(f"Invalid rank: {self.rank}. Must be between 2 and 14.")
        if self.suit not in ['h', 'd', 'c', 's']:
            raise ValueError(f"Invalid suit: {self.suit}. Must be one of 'h', 'd', 'c', 's'.")
    
    def __str__(self) -> str:
        """String representation of the card."""
        rank_str = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(self.rank, str(self.rank))
        suit_str = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}[self.suit]
        return f"{rank_str}{suit_str}"
    
    def __eq__(self, other) -> bool:
        """Check equality of cards."""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        """Hash function for card."""
        return hash((self.rank, self.suit))


@dataclass
class Action:
    """表示扑克中的玩家动作。
    
    Attributes:
        action_type: 动作类型（FOLD, CHECK, CALL, RAISE, RAISE_SMALL, RAISE_BIG, ALL_IN）
        amount: 加注金额（仅用于加注动作和全下动作）
    """
    action_type: ActionType
    amount: int = 0  # 仅用于加注动作和全下动作
    
    # 所有加注类型的集合（不包括 ALL_IN，因为 ALL_IN 是特殊情况）
    RAISE_TYPES = {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG}
    
    # 需要金额的动作类型
    AMOUNT_TYPES = {ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN}
    
    def __post_init__(self):
        """验证动作值。"""
        if self.action_type in self.AMOUNT_TYPES and self.amount <= 0:
            raise ValueError(f"{self.action_type.value} 动作必须有正的金额，得到 {self.amount}")
        if self.action_type not in self.AMOUNT_TYPES and self.amount != 0:
            raise ValueError(f"{self.action_type.value} 动作的金额应为0")


@dataclass
class GameState:
    """Represents the complete state of a poker game.
    
    Attributes:
        player_hands: List of player hands (each hand is a tuple of 2 cards)
        community_cards: List of community cards (0-5 cards)
        pot: Current pot size
        player_stacks: List of player chip stacks
        current_bets: List of current bets for each player
        button_position: Position of the dealer button (0 or 1)
        stage: Current game stage
        action_history: History of actions taken
        current_player: Index of player whose turn it is (0 or 1)
    """
    player_hands: List[Tuple[Card, Card]]
    community_cards: List[Card]
    pot: int
    player_stacks: List[int]
    current_bets: List[int]
    button_position: int
    stage: GameStage
    action_history: List[Action] = field(default_factory=list)
    current_player: int = 0
    
    def __post_init__(self):
        """Validate game state."""
        if len(self.player_hands) != 2:
            raise ValueError(f"Must have exactly 2 players, got {len(self.player_hands)}")
        if len(self.player_stacks) != 2:
            raise ValueError(f"Must have exactly 2 player stacks, got {len(self.player_stacks)}")
        if len(self.current_bets) != 2:
            raise ValueError(f"Must have exactly 2 current bets, got {len(self.current_bets)}")
        if self.button_position not in [0, 1]:
            raise ValueError(f"Button position must be 0 or 1, got {self.button_position}")
        if self.current_player not in [0, 1]:
            raise ValueError(f"Current player must be 0 or 1, got {self.current_player}")
        if len(self.community_cards) > 5:
            raise ValueError(f"Cannot have more than 5 community cards, got {len(self.community_cards)}")
        if self.pot < 0:
            raise ValueError(f"Pot cannot be negative, got {self.pot}")
        for stack in self.player_stacks:
            if stack < 0:
                raise ValueError(f"Player stack cannot be negative, got {stack}")


@dataclass
class Episode:
    """Represents a complete training episode (one poker hand).
    
    Attributes:
        states: Sequence of game states
        actions: Sequence of actions taken
        rewards: Sequence of rewards received
        player_id: ID of the player (0 or 1)
        final_reward: Final reward at end of episode
    """
    states: List[GameState]
    actions: List[Action]
    rewards: List[float]
    player_id: int
    final_reward: float
    
    def __post_init__(self):
        """Validate episode data."""
        if self.player_id not in [0, 1]:
            raise ValueError(f"Player ID must be 0 or 1, got {self.player_id}")
        if len(self.states) != len(self.actions) + 1:
            raise ValueError(
                f"Number of states ({len(self.states)}) must be one more than "
                f"number of actions ({len(self.actions)})"
            )
        if len(self.actions) != len(self.rewards):
            raise ValueError(
                f"Number of actions ({len(self.actions)}) must equal "
                f"number of rewards ({len(self.rewards)})"
            )


@dataclass
class TrainingConfig:
    """训练配置参数。
    
    Attributes:
        learning_rate: 神经网络优化的学习率
        batch_size: 训练批次大小
        num_episodes: 总训练回合数
        discount_factor: 未来奖励的折扣因子（gamma）
        network_architecture: 隐藏层维度列表
        checkpoint_interval: 检查点保存间隔（回合数）
        num_parallel_envs: 并行游戏环境数量
        initial_stack: 每个玩家的初始筹码
        small_blind: 小盲注金额
        big_blind: 大盲注金额
        entropy_coefficient: 熵正则化系数（防止策略坍塌）
        max_raises_per_street: 每条街最大加注次数（0=无限制）
        regret_buffer_size: 遗憾缓冲区大小（Deep CFR）
        strategy_buffer_size: 策略缓冲区大小（Deep CFR）
        cfr_iterations_per_update: 每次网络更新前的 CFR 迭代次数
        network_train_steps: 每次更新的训练步数
        use_abstraction: 是否启用卡牌抽象
        abstraction_path: 预计算抽象文件路径（目录路径）
        abstraction_config: 抽象配置参数（字典格式，用于创建AbstractionConfig）
        cfr_variant: CFR变体类型（standard, cfr_plus, lcfr, dcfr）
        regret_decay_factor: 遗憾值衰减因子（0-1之间，默认0.99）
        regret_clip_threshold: 遗憾值裁剪阈值（正数，默认100.0）
        use_huber_loss: 是否使用Huber损失替代MSE损失
        ema_decay: EMA目标网络衰减率（0-1之间，默认0.995）
        gradient_clip_norm: 梯度裁剪范数（正数，默认1.0）
    """
    learning_rate: float = 0.001
    batch_size: int = 32
    num_episodes: int = 10000
    discount_factor: float = 0.99
    network_architecture: List[int] = field(default_factory=lambda: [512, 256, 128])
    checkpoint_interval: int = 1000
    num_parallel_envs: int = 1
    initial_stack: int = 1000
    small_blind: int = 5
    big_blind: int = 10
    entropy_coefficient: float = 0.01  # 熵正则化系数，防止策略坍塌
    max_raises_per_street: int = 4  # 每条街最大加注次数
    # Deep CFR 特有参数
    regret_buffer_size: int = 2000000  # 遗憾缓冲区大小
    strategy_buffer_size: int = 2000000  # 策略缓冲区大小
    cfr_iterations_per_update: int = 1000  # 每次网络更新前的 CFR 迭代次数
    network_train_steps: int = 4000  # 每次更新的训练步数
    # 卡牌抽象参数
    use_abstraction: bool = False  # 是否启用卡牌抽象
    abstraction_path: str = ""  # 预计算抽象文件路径（目录路径）
    abstraction_config: Dict[str, Any] = field(default_factory=dict)  # 抽象配置参数
    # 收敛控制参数（需求: 1.2, 1.3, 1.4, 3.1, 3.3, 6.1）
    cfr_variant: str = "cfr_plus"  # CFR变体类型：standard, cfr_plus, lcfr, dcfr
    regret_decay_factor: float = 0.99  # 遗憾值衰减因子（需求: 1.2）
    regret_clip_threshold: float = 100.0  # 遗憾值裁剪阈值（需求: 1.3）
    use_huber_loss: bool = True  # 是否使用Huber损失（需求: 1.4）
    ema_decay: float = 0.995  # EMA目标网络衰减率（需求: 3.1）
    gradient_clip_norm: float = 1.0  # 梯度裁剪范数（需求: 3.3）
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.num_episodes <= 0:
            raise ValueError(f"Number of episodes must be positive, got {self.num_episodes}")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError(f"Discount factor must be in [0, 1], got {self.discount_factor}")
        if self.checkpoint_interval <= 0:
            raise ValueError(f"Checkpoint interval must be positive, got {self.checkpoint_interval}")
        if self.num_parallel_envs <= 0:
            raise ValueError(f"Number of parallel environments must be positive, got {self.num_parallel_envs}")
        if self.initial_stack <= 0:
            raise ValueError(f"Initial stack must be positive, got {self.initial_stack}")
        if self.small_blind <= 0:
            raise ValueError(f"Small blind must be positive, got {self.small_blind}")
        if self.big_blind <= 0:
            raise ValueError(f"Big blind must be positive, got {self.big_blind}")
        if self.big_blind <= self.small_blind:
            raise ValueError(f"Big blind ({self.big_blind}) must be greater than small blind ({self.small_blind})")
        if self.entropy_coefficient < 0:
            raise ValueError(f"Entropy coefficient must be non-negative, got {self.entropy_coefficient}")
        # Deep CFR 参数验证
        if self.regret_buffer_size <= 0:
            raise ValueError(f"Regret buffer size must be positive, got {self.regret_buffer_size}")
        if self.strategy_buffer_size <= 0:
            raise ValueError(f"Strategy buffer size must be positive, got {self.strategy_buffer_size}")
        if self.cfr_iterations_per_update <= 0:
            raise ValueError(f"CFR iterations per update must be positive, got {self.cfr_iterations_per_update}")
        if self.network_train_steps <= 0:
            raise ValueError(f"Network train steps must be positive, got {self.network_train_steps}")
        # 卡牌抽象参数验证
        if not isinstance(self.use_abstraction, bool):
            raise ValueError(f"use_abstraction must be a boolean, got {type(self.use_abstraction).__name__}")
        if not isinstance(self.abstraction_path, str):
            raise ValueError(f"abstraction_path must be a string, got {type(self.abstraction_path).__name__}")
        if not isinstance(self.abstraction_config, dict):
            raise ValueError(f"abstraction_config must be a dict, got {type(self.abstraction_config).__name__}")
        # 如果启用抽象但未提供路径，验证抽象配置参数
        if self.use_abstraction and self.abstraction_config:
            self._validate_abstraction_config(self.abstraction_config)
        # 收敛控制参数验证（需求: 1.2, 1.3, 1.4, 3.1, 3.3, 6.1）
        self._validate_convergence_control_config()
    
    def _validate_abstraction_config(self, config: Dict[str, Any]) -> None:
        """验证抽象配置参数。
        
        Args:
            config: 抽象配置字典
            
        Raises:
            ValueError: 如果配置参数无效
        """
        # 验证桶数量参数
        bucket_params = ['preflop_buckets', 'flop_buckets', 'turn_buckets', 'river_buckets']
        for param in bucket_params:
            if param in config:
                value = config[param]
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"abstraction_config.{param} must be a positive integer, got {value}")
        
        # 验证翻牌前桶数不超过169
        if 'preflop_buckets' in config and config['preflop_buckets'] > 169:
            raise ValueError(f"abstraction_config.preflop_buckets cannot exceed 169, got {config['preflop_buckets']}")
        
        # 验证equity_bins参数
        if 'equity_bins' in config:
            value = config['equity_bins']
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"abstraction_config.equity_bins must be a positive integer, got {value}")
        
        # 验证kmeans参数
        if 'kmeans_restarts' in config:
            value = config['kmeans_restarts']
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"abstraction_config.kmeans_restarts must be a positive integer, got {value}")
        
        if 'kmeans_max_iters' in config:
            value = config['kmeans_max_iters']
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"abstraction_config.kmeans_max_iters must be a positive integer, got {value}")
        
        # 验证布尔参数
        if 'use_potential_aware' in config:
            if not isinstance(config['use_potential_aware'], bool):
                raise ValueError(f"abstraction_config.use_potential_aware must be a boolean")
    
    def _validate_convergence_control_config(self) -> None:
        """验证收敛控制配置参数。
        
        验证CFR变体、遗憾值衰减、裁剪阈值、Huber损失、EMA衰减和梯度裁剪等参数。
        
        Raises:
            ValueError: 如果配置参数无效
        """
        # 验证CFR变体类型（需求: 6.1）
        valid_cfr_variants = ["standard", "cfr_plus", "lcfr", "dcfr"]
        if self.cfr_variant not in valid_cfr_variants:
            raise ValueError(
                f"cfr_variant必须是{valid_cfr_variants}之一，当前值: {self.cfr_variant}"
            )
        
        # 验证遗憾值衰减因子（需求: 1.2）
        if not 0 < self.regret_decay_factor <= 1:
            raise ValueError(
                f"regret_decay_factor必须在(0, 1]范围内，当前值: {self.regret_decay_factor}"
            )
        
        # 验证遗憾值裁剪阈值（需求: 1.3）
        if self.regret_clip_threshold <= 0:
            raise ValueError(
                f"regret_clip_threshold必须为正数，当前值: {self.regret_clip_threshold}"
            )
        
        # 验证Huber损失开关（需求: 1.4）
        if not isinstance(self.use_huber_loss, bool):
            raise ValueError(
                f"use_huber_loss必须是布尔值，当前类型: {type(self.use_huber_loss).__name__}"
            )
        
        # 验证EMA衰减率（需求: 3.1）
        if not 0 < self.ema_decay < 1:
            raise ValueError(
                f"ema_decay必须在(0, 1)范围内，当前值: {self.ema_decay}"
            )
        
        # 验证梯度裁剪范数（需求: 3.3）
        if self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm必须为正数，当前值: {self.gradient_clip_norm}"
            )
    
    def get_cfr_variant_type(self) -> 'CFRVariantType':
        """获取CFR变体类型枚举值。
        
        Returns:
            CFRVariantType枚举值
        """
        return CFRVariantType(self.cfr_variant)


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint.
    
    Attributes:
        path: File path to the checkpoint
        episode_number: Number of episodes completed when checkpoint was saved
        timestamp: Time when checkpoint was saved
        win_rate: Win rate at time of checkpoint
        avg_reward: Average reward at time of checkpoint
    """
    path: str
    episode_number: int
    timestamp: datetime
    win_rate: float
    avg_reward: float
    
    def __post_init__(self):
        """Validate checkpoint info."""
        if self.episode_number < 0:
            raise ValueError(f"Episode number cannot be negative, got {self.episode_number}")
        if not 0 <= self.win_rate <= 1:
            raise ValueError(f"Win rate must be in [0, 1], got {self.win_rate}")
