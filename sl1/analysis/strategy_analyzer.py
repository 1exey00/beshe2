"""策略分析器模块 - 分析和可视化训练好的AI策略。

本模块实现了策略分析功能：
- 从检查点加载模型（支持新旧两种格式）
- 分析特定状态下的行动概率分布
- 生成手牌范围的策略热图
- 解释决策（包含期望价值计算）
- 比较多个模型的策略

支持的检查点格式：
- Deep CFR格式（新）：包含regret_network和policy_network
- 旧格式：包含policy_network和value_network（兼容性处理）
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import numpy as np
import torch
import torch.nn as nn

from models.core import GameState, Action, ActionType, GameStage, Card
from models.networks import PolicyNetwork, RegretNetwork
from environment.state_encoder import StateEncoder
from environment.poker_environment import PokerEnvironment
from utils.checkpoint_manager import CheckpointManager

# 延迟导入以避免循环导入
# ActionConfig 在需要时从 viewer.models 导入
def _get_action_config_class():
    """延迟获取ActionConfig类以避免循环导入。"""
    from viewer.models import ActionConfig
    return ActionConfig


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class ActionProbability:
    """行动概率信息。
    
    Attributes:
        action_type: 行动类型
        amount: 加注金额（仅用于RAISE）
        probability: 概率值
    """
    action_type: str
    amount: int
    probability: float
    
    def to_dict(self) -> Dict:
        """转换为字典。"""
        return asdict(self)


@dataclass
class DecisionExplanation:
    """决策解释信息。
    
    Attributes:
        state_description: 状态描述
        action_probabilities: 各行动的概率
        recommended_action: 推荐行动
        expected_value: 期望价值
        reasoning: 决策理由
    """
    state_description: str
    action_probabilities: List[ActionProbability]
    recommended_action: str
    expected_value: float
    reasoning: str
    
    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            'state_description': self.state_description,
            'action_probabilities': [ap.to_dict() for ap in self.action_probabilities],
            'recommended_action': self.recommended_action,
            'expected_value': self.expected_value,
            'reasoning': self.reasoning
        }


@dataclass
class StrategyComparison:
    """策略比较结果。
    
    Attributes:
        models: 模型名称列表
        state_description: 状态描述
        strategies: 每个模型的策略 {模型名: {行动: 概率}}
        timestamp: 比较时间
    """
    models: List[str]
    state_description: str
    strategies: Dict[str, Dict[str, float]]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """转换为字典。"""
        return asdict(self)


# ============================================================================
# 策略分析器类
# ============================================================================

class StrategyAnalyzer:
    """策略分析器 - 分析和可视化训练好的AI策略。
    
    提供以下功能：
    - 从检查点加载模型
    - 分析特定状态下的行动概率分布
    - 生成手牌范围的策略热图
    - 解释决策（包含期望价值计算）
    - 比较多个模型的策略
    - 支持动态动作配置
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
        action_config: Optional[Any] = None
    ):
        """初始化策略分析器。
        
        Args:
            checkpoint_dir: 检查点目录
            device: 计算设备（cpu或cuda）
            action_config: 动作配置（可选，如果不提供将在加载模型时自动检测）
                类型为 viewer.models.ActionConfig
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.encoder = StateEncoder()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 当前加载的模型（Deep CFR架构）
        self._policy_network: Optional[PolicyNetwork] = None
        self._regret_network: Optional[RegretNetwork] = None
        self._model_metadata: Dict[str, Any] = {}
        self._checkpoint_format: str = 'unknown'  # 'deep_cfr_v1' 或 'legacy'
        
        # 动作配置（支持动态动作空间）
        # 类型为 viewer.models.ActionConfig
        self._action_config: Optional[Any] = action_config
        
        # 显示用的动作列表（合并CHECK和CALL为CHECK/CALL）
        self._display_actions: List[str] = []
        if action_config is not None:
            self._display_actions = self._build_display_actions()
    
    def load_model(
        self,
        checkpoint_path: Union[str, Path],
        input_dim: int = 370,
        hidden_dims: List[int] = None,
        action_dim: int = None
    ) -> None:
        """从检查点加载模型。
        
        支持两种检查点格式：
        1. Deep CFR格式（新）：包含regret_network和policy_network
        2. 旧格式：包含policy_network和value_network（兼容性处理）
        
        自动检测模型的 action_dim（4 或 5）以支持向后兼容。
        
        Args:
            checkpoint_path: 检查点文件路径
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            action_dim: 行动空间维度（如果为 None，自动检测）
            
        Raises:
            FileNotFoundError: 检查点文件不存在
            RuntimeError: 模型加载失败
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 加载检查点数据
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        
        # 检测检查点格式
        checkpoint_format = checkpoint_data.get('checkpoint_format', 'legacy')
        self._checkpoint_format = checkpoint_format
        
        # 自动检测 action_dim
        if action_dim is None:
            action_dim = self._detect_action_dim(checkpoint_data)
        
        # 创建策略网络
        self._policy_network = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim
        ).to(self.device)
        
        if checkpoint_format == 'deep_cfr_v1':
            # 新格式：Deep CFR
            self._load_deep_cfr_checkpoint(checkpoint_data, input_dim, hidden_dims, action_dim)
        else:
            # 旧格式：兼容性处理
            self._load_legacy_checkpoint(checkpoint_data)
        
        self._policy_network.eval()
        if self._regret_network is not None:
            self._regret_network.eval()
        
        # 提取元数据
        self._model_metadata = self._extract_metadata(checkpoint_data)
        
        # 设置动作配置（从检查点或根据维度使用默认配置）
        ActionConfig = _get_action_config_class()
        self._action_config = ActionConfig.from_checkpoint(checkpoint_data)
        self._display_actions = self._build_display_actions()
    
    def _detect_action_dim(self, checkpoint_data: Dict[str, Any]) -> int:
        """自动检测检查点的 action_dim。
        
        通过检查网络权重的输出层维度来确定 action_dim。
        
        Args:
            checkpoint_data: 检查点数据字典
            
        Returns:
            action_dim（4 或 5）
        """
        # 尝试从策略网络的输出层检测
        if 'policy_network_state_dict' in checkpoint_data:
            state_dict = checkpoint_data['policy_network_state_dict']
            # 查找输出层的权重
            for key in state_dict:
                if 'network' in key and 'weight' in key:
                    # 获取最后一层的输出维度
                    pass
            # 查找最后一个线性层
            last_weight_key = None
            for key in sorted(state_dict.keys()):
                if 'weight' in key:
                    last_weight_key = key
            if last_weight_key:
                return state_dict[last_weight_key].shape[0]
        
        # 尝试从旧格式的 model_state_dict 检测
        if 'model_state_dict' in checkpoint_data:
            state_dict = checkpoint_data['model_state_dict']
            last_weight_key = None
            for key in sorted(state_dict.keys()):
                if 'weight' in key:
                    last_weight_key = key
            if last_weight_key:
                return state_dict[last_weight_key].shape[0]
        
        # 默认返回 5（新格式）
        return 5
    
    def _load_deep_cfr_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        input_dim: int,
        hidden_dims: List[int],
        action_dim: int
    ) -> None:
        """加载Deep CFR格式的检查点。
        
        Args:
            checkpoint_data: 检查点数据字典
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            action_dim: 行动空间维度
        """
        # 加载策略网络
        if 'policy_network_state_dict' in checkpoint_data:
            self._policy_network.load_state_dict(checkpoint_data['policy_network_state_dict'])
        
        # 创建并加载遗憾网络
        self._regret_network = RegretNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim
        ).to(self.device)
        
        if 'regret_network_state_dict' in checkpoint_data:
            self._regret_network.load_state_dict(checkpoint_data['regret_network_state_dict'])
    
    def _load_legacy_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """加载旧格式的检查点（兼容性处理）。
        
        旧格式包含policy_network和value_network。
        我们只加载policy_network，遗憾网络设为None。
        
        Args:
            checkpoint_data: 检查点数据字典
        """
        # 加载策略网络（旧格式中的model_state_dict）
        if 'model_state_dict' in checkpoint_data:
            self._policy_network.load_state_dict(checkpoint_data['model_state_dict'])
        
        # 旧格式没有遗憾网络
        self._regret_network = None
    
    def _extract_metadata(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """从检查点数据中提取元数据。
        
        Args:
            checkpoint_data: 检查点数据字典
            
        Returns:
            元数据字典
        """
        metadata = {
            'episode_number': checkpoint_data.get('episode_number', 0),
            'timestamp': checkpoint_data.get('timestamp', ''),
            'win_rate': checkpoint_data.get('win_rate', 0.0),
            'avg_reward': checkpoint_data.get('avg_reward', 0.0),
            'checkpoint_format': checkpoint_data.get('checkpoint_format', 'legacy'),
        }
        
        # 添加Deep CFR特有的元数据
        if metadata['checkpoint_format'] == 'deep_cfr_v1':
            metadata['cfr_iterations'] = checkpoint_data.get('cfr_iterations', 0)
            metadata['regret_buffer_size'] = checkpoint_data.get('regret_buffer_size', 0)
            metadata['strategy_buffer_size'] = checkpoint_data.get('strategy_buffer_size', 0)
        
        # 添加其他元数据
        excluded_keys = {
            'model_state_dict', 'optimizer_state_dict',
            'value_network_state_dict', 'value_optimizer_state_dict',
            'regret_network_state_dict', 'policy_network_state_dict',
            'regret_optimizer_state_dict', 'policy_optimizer_state_dict',
            'episode_number', 'timestamp', 'win_rate', 'avg_reward',
            'has_value_network', 'checkpoint_format'
        }
        for key, value in checkpoint_data.items():
            if key not in excluded_keys and key not in metadata:
                metadata[key] = value
        
        return metadata
    
    def _build_display_actions(self) -> List[str]:
        """构建显示用的动作列表。
        
        将CHECK和CALL合并为CHECK/CALL显示。
        
        Returns:
            显示用的动作名称列表
        """
        if self._action_config is None:
            return ['FOLD', 'CHECK/CALL', 'RAISE_SMALL', 'RAISE_BIG']
        
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
    
    def set_action_config(self, config: Any) -> None:
        """设置动作配置。
        
        Args:
            config: 动作配置对象（viewer.models.ActionConfig类型）
        """
        self._action_config = config
        self._display_actions = self._build_display_actions()
    
    @property
    def action_config(self) -> Optional[Any]:
        """获取当前动作配置。
        
        Returns:
            动作配置对象（viewer.models.ActionConfig类型），如果未设置则返回None
        """
        return self._action_config
    
    @property
    def available_actions(self) -> List[str]:
        """获取可用动作列表（显示用）。
        
        Returns:
            显示用的动作名称列表
        """
        return self._display_actions.copy()
    
    def analyze_state(
        self,
        state: GameState,
        player_id: int = 0,
        filter_illegal: bool = True,
        use_regret_network: bool = False
    ) -> Dict[str, float]:
        """分析特定状态下的行动概率分布。
        
        根据游戏状态过滤不合法的行动：
        - 当可以过牌（CHECK）时，FOLD 不合法
        - 当需要跟注时，CHECK 不合法
        
        使用动态动作配置，返回的动作数量与模型的动作维度一致。
        
        Args:
            state: 游戏状态
            player_id: 玩家ID（0或1）
            filter_illegal: 是否过滤不合法的行动
            use_regret_network: 是否使用后悔值网络计算策略（通过Regret Matching）
                - True: 使用后悔值网络的遗憾值，先过滤到合法动作再进行Regret Matching
                - False: 使用策略网络的get_action_probs()方法（平均策略）
            
        Returns:
            行动概率分布字典 {行动名称: 概率}
            动作数量与模型的动作维度一致
            
        Raises:
            RuntimeError: 模型未加载
        """
        if self._policy_network is None:
            raise RuntimeError("模型未加载，请先调用load_model方法")
        
        with torch.no_grad():
            # 编码状态
            encoding = self.encoder.encode(state, player_id)
            state_tensor = torch.tensor(
                encoding, dtype=torch.float32
            ).to(self.device)
            
            # 获取行动概率
            if use_regret_network and self._regret_network is not None:
                # 使用后悔值网络，先获取遗憾值，然后过滤到合法动作再进行Regret Matching
                regrets = self._regret_network(state_tensor).cpu().numpy()
                probs = self._regret_matching_with_legal_filter(regrets, state)
            else:
                # 使用策略网络
                probs = self._policy_network.get_action_probs(state_tensor)
                probs = probs.cpu().numpy()
        
        # 使用动态动作配置
        return self._convert_probs_to_display_format(probs, state, filter_illegal)
    
    def _regret_matching_with_legal_filter(
        self,
        regrets: np.ndarray,
        state: GameState
    ) -> np.ndarray:
        """对遗憾值进行Regret Matching，同时过滤非法动作。
        
        在进行Regret Matching之前，先将非法动作的遗憾值设为负无穷，
        这样它们在取正遗憾值时会变成0。
        
        Args:
            regrets: 遗憾值数组
            state: 游戏状态
            
        Returns:
            策略概率数组
        """
        # 检查当前状态下哪些行动是合法的
        bets_equal = state.current_bets[0] == state.current_bets[1]
        
        # 获取动作配置
        action_names = self._action_config.action_names if self._action_config else []
        
        # 创建合法动作掩码
        legal_mask = np.ones(len(regrets), dtype=bool)
        
        for i, name in enumerate(action_names):
            if i >= len(regrets):
                break
            if name == 'FOLD' and bets_equal:
                # 下注相等时不能弃牌
                legal_mask[i] = False
            elif name == 'CHECK' and not bets_equal:
                # 下注不等时不能过牌
                legal_mask[i] = False
            elif name == 'CALL' and bets_equal:
                # 下注相等时不能跟注（应该用CHECK）
                legal_mask[i] = False
        
        # 将非法动作的遗憾值设为负无穷
        filtered_regrets = regrets.copy()
        filtered_regrets[~legal_mask] = -np.inf
        
        # 取正遗憾值
        positive_regrets = np.maximum(filtered_regrets, 0.0)
        
        # 计算正遗憾值的和
        regret_sum = positive_regrets.sum()
        
        # 如果正遗憾值和 > 0，按比例分配；否则使用均匀分布（只在合法动作上）
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            # 均匀分布在合法动作上
            num_legal = legal_mask.sum()
            strategy = np.zeros_like(regrets)
            if num_legal > 0:
                strategy[legal_mask] = 1.0 / num_legal
            else:
                # 如果没有合法动作，返回均匀分布
                strategy = np.ones_like(regrets) / len(regrets)
        
        return strategy
    
    def _convert_probs_to_display_format(
        self,
        probs: np.ndarray,
        state: GameState,
        filter_illegal: bool
    ) -> Dict[str, float]:
        """将网络输出的概率转换为显示格式。
        
        根据动作配置动态处理不同维度的动作空间。
        
        Args:
            probs: 网络输出的概率数组
            state: 游戏状态
            filter_illegal: 是否过滤不合法的行动
            
        Returns:
            显示格式的行动概率字典
        """
        # 检查当前状态下哪些行动是合法的
        bets_equal = state.current_bets[0] == state.current_bets[1]
        
        # 获取动作配置
        action_names = self._action_config.action_names if self._action_config else []
        action_dim = len(probs)
        
        # 构建原始概率字典
        raw_probs = {}
        for i, prob in enumerate(probs):
            if i < len(action_names):
                raw_probs[action_names[i]] = float(prob)
            else:
                raw_probs[f'ACTION_{i}'] = float(prob)
        
        # 转换为显示格式（合并CHECK和CALL为CHECK/CALL）
        result = {}
        
        # 处理FOLD
        if 'FOLD' in raw_probs:
            if filter_illegal and bets_equal:
                result['FOLD'] = 0.0
            else:
                result['FOLD'] = raw_probs['FOLD']
        
        # 处理CHECK/CALL合并
        check_prob = raw_probs.get('CHECK', 0.0)
        call_prob = raw_probs.get('CALL', 0.0)
        
        if 'CHECK' in raw_probs or 'CALL' in raw_probs:
            if filter_illegal:
                if bets_equal:
                    # 可以过牌，使用CHECK概率
                    result['CHECK/CALL'] = check_prob
                else:
                    # 需要跟注，使用CALL概率
                    result['CHECK/CALL'] = call_prob
            else:
                # 不过滤，合并CHECK和CALL概率
                result['CHECK/CALL'] = check_prob + call_prob
        
        # 处理其他动作（RAISE_SMALL, RAISE_BIG, ALL_IN等）
        for action in action_names:
            if action not in ['FOLD', 'CHECK', 'CALL']:
                if action in raw_probs:
                    result[action] = raw_probs[action]
        
        # 处理旧模型（4维输出）的RAISE动作
        if action_dim == 4 and 'RAISE' in raw_probs:
            # 将RAISE概率平均分配给RAISE_SMALL和RAISE_BIG
            raise_prob = raw_probs['RAISE']
            result['RAISE_SMALL'] = raise_prob * 0.5
            result['RAISE_BIG'] = raise_prob * 0.5
        
        # 归一化概率
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        else:
            # 如果总和为0，返回均匀分布
            num_actions = len(result)
            if num_actions > 0:
                result = {k: 1.0 / num_actions for k in result}
        
        return result
    
    def generate_strategy_heatmap(
        self,
        hand_range: List[Tuple[Card, Card]],
        community_cards: List[Card] = None,
        stage: GameStage = GameStage.PREFLOP,
        player_id: int = 0,
        pot: int = 100,
        player_stacks: List[int] = None
    ) -> np.ndarray:
        """生成手牌范围的策略热图。
        
        Args:
            hand_range: 手牌组合列表
            community_cards: 公共牌（可选）
            stage: 游戏阶段
            player_id: 玩家ID
            pot: 底池大小
            player_stacks: 玩家筹码
            
        Returns:
            热图数据数组，形状为(len(hand_range), num_actions)
            每行对应一个手牌组合，每列对应一个行动的概率
            
        Raises:
            RuntimeError: 模型未加载
        """
        if self._policy_network is None:
            raise RuntimeError("模型未加载，请先调用load_model方法")
        
        if community_cards is None:
            community_cards = []
        
        if player_stacks is None:
            player_stacks = [1000, 1000]
        
        num_hands = len(hand_range)
        num_actions = self._policy_network.action_dim
        heatmap = np.zeros((num_hands, num_actions), dtype=np.float32)
        
        for i, hand in enumerate(hand_range):
            # 创建游戏状态
            # 为对手创建一个虚拟手牌（不影响策略分析）
            opponent_hand = self._create_dummy_hand(hand, community_cards)
            
            if player_id == 0:
                player_hands = [hand, opponent_hand]
            else:
                player_hands = [opponent_hand, hand]
            
            state = GameState(
                player_hands=player_hands,
                community_cards=community_cards,
                pot=pot,
                player_stacks=player_stacks.copy(),
                current_bets=[0, 0],
                button_position=0,
                stage=stage,
                action_history=[],
                current_player=player_id
            )
            
            # 获取行动概率
            probs = self.analyze_state(state, player_id)
            
            # 将显示格式的概率映射到原始行动索引
            # 使用动态动作配置
            action_names = self._action_config.action_names if self._action_config else []
            
            for j, action_name in enumerate(action_names):
                if j >= num_actions:
                    break
                    
                if action_name == 'CHECK' or action_name == 'CALL':
                    # CHECK和CALL在显示格式中合并为CHECK/CALL
                    heatmap[i, j] = probs.get('CHECK/CALL', 0.0) * 0.5
                elif action_name in probs:
                    heatmap[i, j] = probs.get(action_name, 0.0)
                else:
                    heatmap[i, j] = 0.0
        
        return heatmap
    
    def explain_decision(
        self,
        state: GameState,
        player_id: int = 0
    ) -> DecisionExplanation:
        """解释决策，包含期望价值计算。
        
        Args:
            state: 游戏状态
            player_id: 玩家ID
            
        Returns:
            决策解释对象
            
        Raises:
            RuntimeError: 模型未加载
        """
        if self._policy_network is None:
            raise RuntimeError("模型未加载，请先调用load_model方法")
        
        # 获取行动概率
        probs = self.analyze_state(state, player_id)
        
        # 创建行动概率列表
        action_probabilities = []
        for action_name, prob in probs.items():
            amount = 0
            if 'RAISE' in action_name:
                amount = state.pot // 2 if 'SMALL' in action_name else state.pot
            
            action_probabilities.append(ActionProbability(
                action_type=action_name,
                amount=amount,
                probability=prob
            ))
        
        # 找到推荐行动（概率最高的）
        recommended_action = max(probs.items(), key=lambda x: x[1])[0]
        
        # 计算期望价值
        expected_value = self._calculate_expected_value(state, player_id, probs)
        
        # 生成状态描述
        state_description = self._describe_state(state, player_id)
        
        # 生成决策理由
        reasoning = self._generate_reasoning(
            state, player_id, probs, recommended_action, expected_value
        )
        
        return DecisionExplanation(
            state_description=state_description,
            action_probabilities=action_probabilities,
            recommended_action=recommended_action,
            expected_value=expected_value,
            reasoning=reasoning
        )
    
    def compare_strategies(
        self,
        checkpoint_paths: Dict[str, Union[str, Path]],
        state: GameState,
        player_id: int = 0,
        input_dim: int = 370,
        hidden_dims: List[int] = None,
        action_dim: int = 5
    ) -> StrategyComparison:
        """比较多个模型的策略。
        
        Args:
            checkpoint_paths: 模型检查点路径字典 {模型名: 路径}
            state: 游戏状态
            player_id: 玩家ID
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            action_dim: 行动空间维度
            
        Returns:
            策略比较结果
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        strategies = {}
        
        for model_name, checkpoint_path in checkpoint_paths.items():
            # 加载模型
            self.load_model(
                checkpoint_path=checkpoint_path,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                action_dim=action_dim
            )
            
            # 分析状态
            probs = self.analyze_state(state, player_id)
            strategies[model_name] = probs
        
        # 生成状态描述
        state_description = self._describe_state(state, player_id)
        
        return StrategyComparison(
            models=list(checkpoint_paths.keys()),
            state_description=state_description,
            strategies=strategies,
            timestamp=datetime.now().isoformat()
        )
    
    def plot_strategy_heatmap(
        self,
        heatmap: np.ndarray,
        hand_labels: List[str] = None,
        action_labels: List[str] = None,
        title: str = "策略热图",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """绘制策略热图。
        
        Args:
            heatmap: 热图数据
            hand_labels: 手牌标签
            action_labels: 行动标签
            title: 图表标题
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
        except ImportError:
            raise ImportError("需要安装matplotlib库来生成可视化")
        
        if action_labels is None:
            # 使用动态动作配置
            if self._action_config:
                action_labels = self._action_config.action_names.copy()
            else:
                action_labels = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG']
        
        if hand_labels is None:
            hand_labels = [f"手牌{i+1}" for i in range(heatmap.shape[0])]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(hand_labels) * 0.3)))
        
        # 绘制热图
        im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(action_labels)))
        ax.set_xticklabels(action_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(hand_labels)))
        ax.set_yticklabels(hand_labels)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='概率')
        
        # 设置标题
        ax.set_title(title)
        ax.set_xlabel('行动')
        ax.set_ylabel('手牌')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.close()
    
    def plot_strategy_comparison(
        self,
        comparison: StrategyComparison,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """绘制策略比较图。
        
        Args:
            comparison: 策略比较结果
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            raise ImportError("需要安装matplotlib库来生成可视化")
        
        models = comparison.models
        # 使用动态动作配置
        if self._action_config:
            actions = self._action_config.action_names.copy()
        else:
            actions = ['FOLD', 'CHECK', 'CALL', 'RAISE_SMALL', 'RAISE_BIG']
        
        # 准备数据
        x = np.arange(len(actions))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model_name in enumerate(models):
            probs = [comparison.strategies[model_name].get(a, 0) for a in actions]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, probs, width, label=model_name)
        
        ax.set_xlabel('行动')
        ax.set_ylabel('概率')
        ax.set_title('策略比较')
        ax.set_xticks(x)
        ax.set_xticklabels(actions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.close()
    
    def save_analysis(
        self,
        result: Union[DecisionExplanation, StrategyComparison],
        filepath: Union[str, Path]
    ) -> str:
        """保存分析结果为JSON文件。
        
        Args:
            result: 分析结果
            filepath: 保存路径
            
        Returns:
            保存的文件路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    # ========================================================================
    # 私有方法
    # ========================================================================
    
    def _create_dummy_hand(
        self,
        exclude_hand: Tuple[Card, Card],
        exclude_community: List[Card]
    ) -> Tuple[Card, Card]:
        """创建一个虚拟手牌（用于策略分析）。
        
        Args:
            exclude_hand: 要排除的手牌
            exclude_community: 要排除的公共牌
            
        Returns:
            虚拟手牌
        """
        excluded = set()
        excluded.add((exclude_hand[0].rank, exclude_hand[0].suit))
        excluded.add((exclude_hand[1].rank, exclude_hand[1].suit))
        for card in exclude_community:
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
    
    def _calculate_expected_value(
        self,
        state: GameState,
        player_id: int,
        probs: Dict[str, float]
    ) -> float:
        """计算期望价值。
        
        使用简化的期望价值计算：
        EV = Σ(P(action) * V(action))
        
        其中V(action)基于行动类型的简化估计：
        - FOLD: -当前下注
        - CHECK/CALL: 0（中性）
        - RAISE: 潜在收益（基于底池大小）
        
        Args:
            state: 游戏状态
            player_id: 玩家ID
            probs: 行动概率
            
        Returns:
            期望价值
        """
        current_bet = state.current_bets[player_id]
        pot = state.pot
        
        # 简化的行动价值估计
        action_values = {
            'FOLD': -current_bet if current_bet > 0 else -state.pot * 0.1,
            'CHECK/CALL': 0,
            'RAISE_SMALL': pot * 0.2,  # 小加注的潜在收益
            'RAISE_BIG': pot * 0.3     # 大加注的潜在收益
        }
        
        expected_value = 0.0
        for action_name, prob in probs.items():
            value = action_values.get(action_name, 0)
            expected_value += prob * value
        
        return expected_value
    
    def _describe_state(self, state: GameState, player_id: int) -> str:
        """生成状态描述。
        
        Args:
            state: 游戏状态
            player_id: 玩家ID
            
        Returns:
            状态描述字符串
        """
        hand = state.player_hands[player_id]
        hand_str = f"{hand[0]} {hand[1]}"
        
        community_str = "无" if not state.community_cards else " ".join(
            str(c) for c in state.community_cards
        )
        
        stage_names = {
            GameStage.PREFLOP: "翻牌前",
            GameStage.FLOP: "翻牌",
            GameStage.TURN: "转牌",
            GameStage.RIVER: "河牌"
        }
        
        position = "按钮位" if state.button_position == player_id else "大盲位"
        
        return (
            f"阶段: {stage_names.get(state.stage, str(state.stage))}\n"
            f"手牌: {hand_str}\n"
            f"公共牌: {community_str}\n"
            f"底池: {state.pot}\n"
            f"筹码: {state.player_stacks[player_id]}\n"
            f"位置: {position}"
        )
    
    def _generate_reasoning(
        self,
        state: GameState,
        player_id: int,
        probs: Dict[str, float],
        recommended_action: str,
        expected_value: float
    ) -> str:
        """生成决策理由。
        
        Args:
            state: 游戏状态
            player_id: 玩家ID
            probs: 行动概率
            recommended_action: 推荐行动
            expected_value: 期望价值
            
        Returns:
            决策理由字符串
        """
        hand = state.player_hands[player_id]
        
        # 分析手牌强度
        hand_strength = self._estimate_hand_strength(hand, state.community_cards)
        
        # 分析位置优势
        position_advantage = "有利" if state.button_position == player_id else "不利"
        
        # 分析底池赔率
        pot_odds = state.pot / max(state.player_stacks[player_id], 1)
        
        reasoning_parts = [
            f"推荐行动: {recommended_action}（概率: {probs[recommended_action]:.2%}）",
            f"期望价值: {expected_value:.2f}",
            f"手牌强度估计: {hand_strength}",
            f"位置: {position_advantage}",
            f"底池赔率: {pot_odds:.2f}"
        ]
        
        # 添加具体建议
        if recommended_action == 'FOLD':
            reasoning_parts.append("建议弃牌，当前手牌在此情况下胜率较低。")
        elif recommended_action == 'CHECK/CALL':
            reasoning_parts.append("建议过牌/跟注，保持在牌局中观察后续发展。")
        elif 'RAISE' in recommended_action:
            reasoning_parts.append("建议加注，当前手牌有较好的价值或诈唬机会。")
        
        return "\n".join(reasoning_parts)
    
    def _estimate_hand_strength(
        self,
        hand: Tuple[Card, Card],
        community_cards: List[Card]
    ) -> str:
        """估计手牌强度。
        
        Args:
            hand: 手牌
            community_cards: 公共牌
            
        Returns:
            手牌强度描述
        """
        card1, card2 = hand
        
        # 检查是否是对子
        if card1.rank == card2.rank:
            if card1.rank >= 10:
                return "强（高对子）"
            elif card1.rank >= 7:
                return "中等（中对子）"
            else:
                return "较弱（小对子）"
        
        # 检查是否同花
        suited = card1.suit == card2.suit
        
        # 检查是否连张
        gap = abs(card1.rank - card2.rank)
        connected = gap == 1
        
        # 检查高牌
        high_card = max(card1.rank, card2.rank)
        
        if high_card >= 12:  # Q或更高
            if suited and connected:
                return "强（同花连张高牌）"
            elif suited or connected:
                return "中上（高牌有潜力）"
            else:
                return "中等（高牌）"
        elif high_card >= 10:
            if suited and connected:
                return "中上（同花连张）"
            elif suited:
                return "中等（同花）"
            else:
                return "中等"
        else:
            if suited and connected:
                return "中等（同花连张）"
            elif suited:
                return "较弱（同花小牌）"
            else:
                return "弱"
    
    @property
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self._policy_network is not None
    
    @property
    def model_metadata(self) -> Dict[str, Any]:
        """获取当前加载模型的元数据。"""
        return self._model_metadata.copy()
    
    @property
    def checkpoint_format(self) -> str:
        """获取当前加载检查点的格式。
        
        Returns:
            'deep_cfr_v1' 表示新的Deep CFR格式
            'legacy' 表示旧格式
            'unknown' 表示未加载模型
        """
        return self._checkpoint_format
    
    @property
    def has_regret_network(self) -> bool:
        """检查是否加载了遗憾网络（仅Deep CFR格式有）。"""
        return self._regret_network is not None
