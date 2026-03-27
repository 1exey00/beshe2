"""神经网络模块：策略网络、遗憾网络和价值网络的实现。

本模块实现了德州扑克AI训练所需的神经网络组件：

Deep CFR 架构（当前使用）：
- RegretNetwork: 遗憾网络，输出每个动作的遗憾值，用于 Regret Matching 生成策略
- PolicyNetwork: 策略网络，输出行动概率分布，学习长期平均策略

旧架构（保留用于兼容性）：
- ValueNetwork: 价值网络，估计状态价值（Actor-Critic 架构）
  - 用于加载旧格式检查点
  - 用于策略查看器的价值估计功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PolicyNetwork(nn.Module):
    """策略网络：输出行动概率分布。
    
    网络架构：
    - 输入层：370维（状态编码）
    - 隐藏层：[512, 256, 128]，使用ReLU激活函数
    - 输出层：行动空间维度（默认5：弃牌、过牌、跟注、小加注、大加注）
    
    Attributes:
        input_dim: 输入维度（默认370）
        hidden_dims: 隐藏层维度列表
        action_dim: 行动空间维度
    """
    
    def __init__(
        self, 
        input_dim: int = 370, 
        hidden_dims: List[int] = None, 
        action_dim: int = 6
    ):
        """初始化策略网络。
        
        Args:
            input_dim: 输入特征维度，默认370
            hidden_dims: 隐藏层维度列表，默认[512, 256, 128]
            action_dim: 行动空间维度，默认6（FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
        """
        super(PolicyNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """前向传播，返回行动logits。
        
        Args:
            state_encoding: 状态编码张量，形状为(batch_size, input_dim)或(input_dim,)
            
        Returns:
            行动logits张量，形状为(batch_size, action_dim)或(action_dim,)
        """
        return self.network(state_encoding)
    
    def get_action_probs(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """获取行动概率分布。
        
        对forward输出应用softmax，返回有效的概率分布。
        
        Args:
            state_encoding: 状态编码张量，形状为(batch_size, input_dim)或(input_dim,)
            
        Returns:
            行动概率分布张量，形状为(batch_size, action_dim)或(action_dim,)
            所有概率非负且和为1
        """
        logits = self.forward(state_encoding)
        return F.softmax(logits, dim=-1)


class RegretNetwork(nn.Module):
    """遗憾网络：预测每个动作的遗憾值。
    
    用于 Deep CFR 算法，学习每个动作的即时遗憾值。
    
    网络架构：
    - 输入层：370维（状态编码）
    - 隐藏层：[512, 256, 128]，使用ReLU激活函数
    - 输出层：行动空间维度（默认5），无激活函数
    
    注意：输出可以是任意实数（正或负），表示遗憾值。
    
    Attributes:
        input_dim: 输入维度（默认370）
        hidden_dims: 隐藏层维度列表
        action_dim: 行动空间维度
    """
    
    def __init__(
        self, 
        input_dim: int = 370, 
        hidden_dims: List[int] = None, 
        action_dim: int = 6
    ):
        """初始化遗憾网络。
        
        Args:
            input_dim: 输入特征维度，默认370
            hidden_dims: 隐藏层维度列表，默认[512, 256, 128]
            action_dim: 行动空间维度，默认6（FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
        """
        super(RegretNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 输出层（无激活函数，遗憾值可以是任意实数）
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """前向传播，返回每个动作的遗憾值。
        
        Args:
            state_encoding: 状态编码张量，形状为(batch_size, input_dim)或(input_dim,)
            
        Returns:
            遗憾值张量，形状为(batch_size, action_dim)或(action_dim,)
            输出可以是任意实数
        """
        return self.network(state_encoding)
    
    def get_strategy(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """使用 Regret Matching 从遗憾值计算策略。
        
        Regret Matching 算法：
        - 取正遗憾值：positive_regrets = max(regrets, 0)
        - 如果正遗憾值和 > 0：strategy = positive_regrets / sum
        - 否则：strategy = uniform distribution
        
        Args:
            state_encoding: 状态编码张量，形状为(batch_size, input_dim)或(input_dim,)
            
        Returns:
            策略概率分布张量，形状为(batch_size, action_dim)或(action_dim,)
            所有概率非负且和为1
        """
        regrets = self.forward(state_encoding)
        return self._regret_matching(regrets)
    
    def _regret_matching(self, regrets: torch.Tensor) -> torch.Tensor:
        """将遗憾值转换为策略概率（Regret Matching）。
        
        Args:
            regrets: 遗憾值张量，形状为(batch_size, action_dim)或(action_dim,)
            
        Returns:
            策略概率分布张量，形状与输入相同
        """
        # 取正遗憾值
        positive_regrets = torch.clamp(regrets, min=0.0)
        
        # 计算正遗憾值的和
        regret_sum = positive_regrets.sum(dim=-1, keepdim=True)
        
        # 创建均匀分布作为默认策略
        uniform = torch.ones_like(regrets) / regrets.shape[-1]
        
        # 如果正遗憾值和 > 0，按比例分配；否则使用均匀分布
        # 使用 where 避免除零
        strategy = torch.where(
            regret_sum > 0,
            positive_regrets / (regret_sum + 1e-10),  # 添加小值避免数值问题
            uniform
        )
        
        return strategy


class ValueNetwork(nn.Module):
    """价值网络：估计状态价值。
    
    注意：这是旧 Actor-Critic 架构的组件，在 Deep CFR 架构中不再用于训练。
    保留此类用于：
    - 加载旧格式检查点（向后兼容性）
    - 策略查看器的价值估计功能
    
    网络架构：
    - 输入层：370维（状态编码）
    - 隐藏层：[512, 256, 128]，使用ReLU激活函数
    - 输出层：1维，使用Tanh激活函数（输出范围[-1, 1]）
    
    Attributes:
        input_dim: 输入维度（默认370）
        hidden_dims: 隐藏层维度列表
    """
    
    def __init__(
        self, 
        input_dim: int = 370, 
        hidden_dims: List[int] = None
    ):
        """初始化价值网络。
        
        Args:
            input_dim: 输入特征维度，默认370
            hidden_dims: 隐藏层维度列表，默认[512, 256, 128]
        """
        super(ValueNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层（单独定义以便应用Tanh）
        self.output_layer = nn.Linear(prev_dim, 1)
    
    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """前向传播，返回状态价值估计。
        
        Args:
            state_encoding: 状态编码张量，形状为(batch_size, input_dim)或(input_dim,)
            
        Returns:
            状态价值估计张量，形状为(batch_size, 1)或(1,)
            输出范围为[-1, 1]
        """
        hidden = self.hidden_layers(state_encoding)
        value = self.output_layer(hidden)
        # 应用Tanh激活函数，将输出限制在[-1, 1]范围内
        return torch.tanh(value)
