"""模型评估器模块 - 评估训练好的AI模型性能。

本模块实现了模型评估功能：
- 运行N局评估对局
- 计算胜率、平均盈利、标准差等指标
- 支持多种对手策略（随机策略、固定策略等）
- 多模型比较
- 评估结果持久化
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from models.core import GameState, Action, ActionType
from environment.poker_environment import PokerEnvironment
from environment.state_encoder import StateEncoder


# ============================================================================
# 对手策略接口
# ============================================================================

class OpponentStrategy(ABC):
    """对手策略的抽象基类。
    
    所有对手策略都必须实现select_action方法。
    """
    
    @abstractmethod
    def select_action(
        self, 
        state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """根据当前状态选择行动。
        
        Args:
            state: 当前游戏状态
            legal_actions: 合法行动列表
            
        Returns:
            选择的行动
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称。"""
        pass


class RandomStrategy(OpponentStrategy):
    """随机策略 - 从合法行动中随机选择。"""
    
    def __init__(self, seed: Optional[int] = None):
        """初始化随机策略。
        
        Args:
            seed: 随机种子（可选）
        """
        self._rng = random.Random(seed)
    
    def select_action(
        self, 
        state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """随机选择一个合法行动。"""
        return self._rng.choice(legal_actions)
    
    @property
    def name(self) -> str:
        return "RandomStrategy"


class FixedStrategy(OpponentStrategy):
    """固定策略 - 按照固定优先级选择行动。
    
    优先级顺序：CHECK > CALL > FOLD（从不主动加注）
    """
    
    def __init__(self, prefer_aggressive: bool = False):
        """初始化固定策略。
        
        Args:
            prefer_aggressive: 是否偏好激进（优先加注）
        """
        self._prefer_aggressive = prefer_aggressive
    
    def select_action(
        self, 
        state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """按固定优先级选择行动。"""
        action_types = {a.action_type: a for a in legal_actions}
        
        if self._prefer_aggressive:
            # 激进策略：优先加注
            raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)
            if any(at in action_types for at in raise_types):
                # 选择最小加注
                raise_actions = [a for a in legal_actions if a.action_type in raise_types]
                return min(raise_actions, key=lambda a: a.amount)
            if ActionType.CALL in action_types:
                return action_types[ActionType.CALL]
            if ActionType.CHECK in action_types:
                return action_types[ActionType.CHECK]
        else:
            # 保守策略：优先过牌/跟注
            if ActionType.CHECK in action_types:
                return action_types[ActionType.CHECK]
            if ActionType.CALL in action_types:
                return action_types[ActionType.CALL]
        
        # 默认弃牌
        return action_types.get(ActionType.FOLD, legal_actions[0])
    
    @property
    def name(self) -> str:
        return f"FixedStrategy({'aggressive' if self._prefer_aggressive else 'passive'})"


class CallOnlyStrategy(OpponentStrategy):
    """只跟注策略 - 只跟注或过牌，从不加注或弃牌。"""
    
    def select_action(
        self, 
        state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """选择跟注或过牌。"""
        action_types = {a.action_type: a for a in legal_actions}
        
        if ActionType.CHECK in action_types:
            return action_types[ActionType.CHECK]
        if ActionType.CALL in action_types:
            return action_types[ActionType.CALL]
        # 如果必须弃牌
        return action_types.get(ActionType.FOLD, legal_actions[0])
    
    @property
    def name(self) -> str:
        return "CallOnlyStrategy"


class AlwaysFoldStrategy(OpponentStrategy):
    """总是弃牌策略 - 除非可以过牌，否则弃牌。"""
    
    def select_action(
        self, 
        state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """选择弃牌或过牌。"""
        action_types = {a.action_type: a for a in legal_actions}
        
        if ActionType.CHECK in action_types:
            return action_types[ActionType.CHECK]
        return action_types.get(ActionType.FOLD, legal_actions[0])
    
    @property
    def name(self) -> str:
        return "AlwaysFoldStrategy"


class NeuralNetworkStrategy(OpponentStrategy):
    """神经网络策略 - 使用训练好的模型做决策。"""
    
    def __init__(
        self, 
        model: nn.Module, 
        encoder: StateEncoder,
        player_id: int = 1,
        device: str = "cpu"
    ):
        """初始化神经网络策略。
        
        Args:
            model: 策略网络模型
            encoder: 状态编码器
            player_id: 玩家ID（0或1）
            device: 计算设备
        """
        self._model = model
        self._encoder = encoder
        self._player_id = player_id
        self._device = device
        self._model.eval()
    
    def select_action(
        self, 
        state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """使用神经网络选择行动。"""
        with torch.no_grad():
            # 编码状态
            encoding = self._encoder.encode(state, self._player_id)
            state_tensor = torch.tensor(encoding, dtype=torch.float32).to(self._device)
            
            # 获取行动概率
            probs = self._model.get_action_probs(state_tensor).cpu().numpy()
            
            # 映射到合法行动
            # 简化映射：0=FOLD, 1=CHECK/CALL, 2=RAISE_SMALL, 3=RAISE_BIG
            action_type_map = {
                ActionType.FOLD: 0,
                ActionType.CHECK: 1,
                ActionType.CALL: 1,
                ActionType.RAISE: 2,  # 向后兼容
                ActionType.RAISE_SMALL: 2,
                ActionType.RAISE_BIG: 3
            }
            
            raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)
            
            # 计算每个合法行动的概率
            legal_probs = []
            for action in legal_actions:
                idx = action_type_map.get(action.action_type, 0)
                if action.action_type in raise_types:
                    # 加注使用索引2或3
                    idx = min(2 + (action.amount > 100), 3)
                legal_probs.append(probs[idx] if idx < len(probs) else 0.1)
            
            # 归一化概率
            total = sum(legal_probs)
            if total > 0:
                legal_probs = [p / total for p in legal_probs]
            else:
                legal_probs = [1.0 / len(legal_actions)] * len(legal_actions)
            
            # 按概率选择行动
            return random.choices(legal_actions, weights=legal_probs, k=1)[0]
    
    @property
    def name(self) -> str:
        return "NeuralNetworkStrategy"


# ============================================================================
# 评估结果数据类
# ============================================================================

@dataclass
class EvaluationResult:
    """单次评估的结果。
    
    Attributes:
        model_name: 模型名称
        opponent_name: 对手策略名称
        num_games: 评估对局数
        wins: 胜局数
        losses: 负局数
        ties: 平局数
        win_rate: 胜率
        avg_profit: 平均盈利
        std_profit: 盈利标准差
        total_profit: 总盈利
        profits: 每局盈利列表
        timestamp: 评估时间
    """
    model_name: str
    opponent_name: str
    num_games: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    avg_profit: float
    std_profit: float
    total_profit: float
    profits: List[float]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """转换为字典。"""
        return asdict(self)


@dataclass
class ComparisonResult:
    """多模型比较结果。
    
    Attributes:
        models: 模型名称列表
        opponent_name: 对手策略名称
        results: 每个模型的评估结果
        timestamp: 比较时间
    """
    models: List[str]
    opponent_name: str
    results: Dict[str, EvaluationResult]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            'models': self.models,
            'opponent_name': self.opponent_name,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'timestamp': self.timestamp
        }


# ============================================================================
# 评估器类
# ============================================================================

class Evaluator:
    """模型评估器 - 评估训练好的AI模型性能。
    
    提供以下功能：
    - 运行N局评估对局
    - 计算胜率、平均盈利、标准差等指标
    - 支持多种对手策略
    - 多模型比较
    - 评估结果持久化
    """
    
    def __init__(
        self,
        initial_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10
    ):
        """初始化评估器。
        
        Args:
            initial_stack: 初始筹码
            small_blind: 小盲注
            big_blind: 大盲注
        """
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.encoder = StateEncoder()
        self._games_played = 0  # 跟踪已执行的对局数
    
    def evaluate(
        self,
        model: nn.Module,
        opponent: OpponentStrategy,
        num_games: int,
        model_name: str = "model",
        model_player_id: int = 0,
        device: str = "cpu"
    ) -> EvaluationResult:
        """运行N局评估对局。
        
        Args:
            model: 要评估的策略网络模型
            opponent: 对手策略
            num_games: 评估对局数
            model_name: 模型名称
            model_player_id: 模型作为哪个玩家（0或1）
            device: 计算设备
            
        Returns:
            评估结果
        """
        model.eval()
        profits = []
        wins = 0
        losses = 0
        ties = 0
        self._games_played = 0
        
        for _ in range(num_games):
            profit, result = self._play_evaluation_game(
                model, opponent, model_player_id, device
            )
            profits.append(profit)
            self._games_played += 1
            
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                ties += 1
        
        # 计算统计指标
        win_rate = wins / num_games if num_games > 0 else 0.0
        avg_profit = float(np.mean(profits)) if profits else 0.0
        std_profit = float(np.std(profits)) if len(profits) > 1 else 0.0
        total_profit = sum(profits)
        
        return EvaluationResult(
            model_name=model_name,
            opponent_name=opponent.name,
            num_games=num_games,
            wins=wins,
            losses=losses,
            ties=ties,
            win_rate=win_rate,
            avg_profit=avg_profit,
            std_profit=std_profit,
            total_profit=total_profit,
            profits=profits,
            timestamp=datetime.now().isoformat()
        )
    
    def _play_evaluation_game(
        self,
        model: nn.Module,
        opponent: OpponentStrategy,
        model_player_id: int,
        device: str
    ) -> Tuple[float, int]:
        """执行单局评估对局。
        
        Args:
            model: 策略网络模型
            opponent: 对手策略
            model_player_id: 模型作为哪个玩家
            device: 计算设备
            
        Returns:
            Tuple[float, int]: (盈利金额, 结果标志: 1=胜, -1=负, 0=平)
        """
        env = PokerEnvironment(
            initial_stack=self.initial_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )
        
        state = env.reset()
        initial_stack = state.player_stacks[model_player_id]
        done = False
        
        while not done:
            current_player = state.current_player
            legal_actions = env.get_legal_actions(state)
            
            # 如果没有合法行动，游戏结束
            if not legal_actions:
                break
            
            if current_player == model_player_id:
                # 模型选择行动
                action = self._select_model_action(
                    model, state, legal_actions, model_player_id, device
                )
            else:
                # 对手选择行动
                action = opponent.select_action(state, legal_actions)
            
            # 如果没有选择到行动，游戏结束
            if action is None:
                break
            
            state, _, done = env.step(action)
        
        # 计算盈利
        final_stack = state.player_stacks[model_player_id]
        profit = final_stack - initial_stack
        
        # 判断胜负
        if profit > 0:
            result = 1
        elif profit < 0:
            result = -1
        else:
            result = 0
        
        return float(profit), result
    
    def _select_model_action(
        self,
        model: nn.Module,
        state: GameState,
        legal_actions: List[Action],
        player_id: int,
        device: str
    ) -> Action:
        """使用模型选择行动。
        
        Args:
            model: 策略网络模型
            state: 当前游戏状态
            legal_actions: 合法行动列表
            player_id: 玩家ID
            device: 计算设备
            
        Returns:
            选择的行动
        """
        # 如果没有合法行动，返回 None
        if not legal_actions:
            return None
        
        with torch.no_grad():
            # 编码状态
            encoding = self.encoder.encode(state, player_id)
            state_tensor = torch.tensor(encoding, dtype=torch.float32).to(device)
            
            # 获取行动概率
            probs = model.get_action_probs(state_tensor).cpu().numpy()
            
            # 映射到合法行动（6维输出：FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN）
            action_type_map = {
                ActionType.FOLD: 0,
                ActionType.CHECK: 1,
                ActionType.CALL: 2,
                ActionType.RAISE: 3,  # 向后兼容
                ActionType.RAISE_SMALL: 3,
                ActionType.RAISE_BIG: 4,
                ActionType.ALL_IN: 5
            }
            
            raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG, ActionType.ALL_IN)
            
            # 计算每个合法行动的概率
            legal_probs = []
            for action in legal_actions:
                idx = action_type_map.get(action.action_type, 0)
                legal_probs.append(probs[idx] if idx < len(probs) else 0.1)
            
            # 归一化概率
            total = sum(legal_probs)
            if total > 0:
                legal_probs = [p / total for p in legal_probs]
            else:
                legal_probs = [1.0 / len(legal_actions)] * len(legal_actions)
            
            # 按概率选择行动
            return random.choices(legal_actions, weights=legal_probs, k=1)[0]
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        opponent: OpponentStrategy,
        num_games: int,
        device: str = "cpu"
    ) -> ComparisonResult:
        """比较多个模型的性能。
        
        Args:
            models: 模型字典 {模型名称: 模型}
            opponent: 对手策略
            num_games: 每个模型的评估对局数
            device: 计算设备
            
        Returns:
            比较结果
        """
        results = {}
        
        for model_name, model in models.items():
            result = self.evaluate(
                model=model,
                opponent=opponent,
                num_games=num_games,
                model_name=model_name,
                device=device
            )
            results[model_name] = result
        
        return ComparisonResult(
            models=list(models.keys()),
            opponent_name=opponent.name,
            results=results,
            timestamp=datetime.now().isoformat()
        )
    
    def save_results(
        self,
        result: Union[EvaluationResult, ComparisonResult],
        filepath: Union[str, Path]
    ) -> str:
        """保存评估结果为JSON文件。
        
        Args:
            result: 评估结果或比较结果
            filepath: 保存路径
            
        Returns:
            保存的文件路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_results(
        self,
        filepath: Union[str, Path]
    ) -> Union[EvaluationResult, ComparisonResult]:
        """从JSON文件加载评估结果。
        
        Args:
            filepath: 文件路径
            
        Returns:
            评估结果或比较结果
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 判断是单个评估结果还是比较结果
        if 'results' in data and isinstance(data['results'], dict):
            # 比较结果
            results = {}
            for model_name, result_data in data['results'].items():
                results[model_name] = EvaluationResult(**result_data)
            
            return ComparisonResult(
                models=data['models'],
                opponent_name=data['opponent_name'],
                results=results,
                timestamp=data['timestamp']
            )
        else:
            # 单个评估结果
            return EvaluationResult(**data)
    
    @property
    def games_played(self) -> int:
        """返回最近一次评估中执行的对局数。"""
        return self._games_played
