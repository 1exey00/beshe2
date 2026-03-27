"""德州扑克AI训练系统的核心数据模型和神经网络。

本模块导出以下组件：

数据模型：
- Card: 扑克牌
- Action: 玩家动作
- GameState: 游戏状态
- Episode: 训练回合
- TrainingConfig: 训练配置
- CheckpointInfo: 检查点信息
- ActionType: 动作类型枚举
- GameStage: 游戏阶段枚举
- HandRank: 手牌等级枚举

神经网络（Deep CFR 架构）：
- RegretNetwork: 遗憾网络，用于 Regret Matching 生成策略
- PolicyNetwork: 策略网络，学习长期平均策略

神经网络（旧架构，保留用于兼容性）：
- ValueNetwork: 价值网络，用于旧检查点加载和策略查看器
"""

from .core import (
    Card,
    Action,
    GameState,
    Episode,
    TrainingConfig,
    CheckpointInfo,
    ActionType,
    GameStage,
    HandRank,
)
from .networks import PolicyNetwork, RegretNetwork, ValueNetwork

__all__ = [
    # 数据模型
    'Card',
    'Action',
    'GameState',
    'Episode',
    'TrainingConfig',
    'CheckpointInfo',
    'ActionType',
    'GameStage',
    'HandRank',
    # Deep CFR 网络
    'RegretNetwork',
    'PolicyNetwork',
    # 旧架构网络（兼容性）
    'ValueNetwork',
]
