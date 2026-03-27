"""配置管理器模块 - 处理训练配置的加载、保存和验证。"""

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from models.core import TrainingConfig


# 默认配置值（与 configs/default_config.json 保持一致）
DEFAULT_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 2048,  # Deep CFR 通常使用较大批次
    'num_episodes': 10000,
    'discount_factor': 1.0,  # CFR 通常不使用折扣
    'network_architecture': [512, 256, 128],
    'checkpoint_interval': 1000,
    'num_parallel_envs': 1,
    'initial_stack': 1000,
    'small_blind': 5,
    'big_blind': 10,
    'entropy_coefficient': 0.01,
    'max_raises_per_street': 4,
    # Deep CFR 特有参数
    'regret_buffer_size': 2000000,
    'strategy_buffer_size': 2000000,
    'cfr_iterations_per_update': 1000,
    'network_train_steps': 4000,
    # 卡牌抽象参数
    'use_abstraction': False,
    'abstraction_path': '',
    'abstraction_config': {}
}

# 旧配置参数（已废弃，用于兼容性处理）
DEPRECATED_PARAMS = ['cfr_weight']

# 必需参数列表（没有默认值的参数）
REQUIRED_PARAMS: List[str] = []  # 当前所有参数都有默认值

# 可选参数列表（有默认值的参数）
OPTIONAL_PARAMS = list(DEFAULT_CONFIG.keys())


class ConfigManager:
    """配置管理器 - 负责训练配置的加载、保存和验证。
    
    提供以下功能：
    - 从JSON文件加载配置
    - 将配置保存为JSON文件
    - 验证配置参数的有效性
    - 为缺失的可选参数应用默认值
    """
    
    def __init__(self):
        """初始化配置管理器。"""
        pass
    
    def load_config(self, path: Union[str, Path]) -> TrainingConfig:
        """从JSON文件加载训练配置。
        
        Args:
            path: JSON配置文件的路径
            
        Returns:
            TrainingConfig: 加载的训练配置对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON格式无效
            ValueError: 配置参数无效
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 应用默认值到缺失的可选参数
        config_dict = self._apply_defaults(config_dict)
        
        # 验证配置
        errors = self.validate_config(config_dict)
        if errors:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
        
        # 创建TrainingConfig对象
        return TrainingConfig(**config_dict)
    
    def save_config(self, config: TrainingConfig, path: Union[str, Path]) -> None:
        """将训练配置保存为JSON文件。
        
        Args:
            config: 要保存的训练配置对象
            path: 保存的目标路径
            
        Raises:
            IOError: 文件写入失败
        """
        path = Path(path)
        
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 将dataclass转换为字典
        config_dict = asdict(config)
        
        # 写入JSON文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def validate_config(self, config: Union[TrainingConfig, Dict[str, Any]]) -> List[str]:
        """验证配置参数的有效性。
        
        Args:
            config: 要验证的配置（可以是TrainingConfig对象或字典）
            
        Returns:
            List[str]: 错误信息列表，如果配置有效则返回空列表
        """
        errors = []
        
        # 如果是TrainingConfig对象，转换为字典
        if isinstance(config, TrainingConfig):
            config_dict = asdict(config)
        else:
            config_dict = config
        
        # 验证learning_rate（学习率）
        if 'learning_rate' in config_dict:
            lr = config_dict['learning_rate']
            if not isinstance(lr, (int, float)):
                errors.append(f"learning_rate: 必须是数值类型，当前类型为 {type(lr).__name__}")
            elif lr <= 0:
                errors.append(f"learning_rate: 必须为正数，当前值为 {lr}")
            elif lr > 1:
                errors.append(f"learning_rate: 不应大于1，当前值为 {lr}")
        
        # 验证batch_size（批次大小）
        if 'batch_size' in config_dict:
            bs = config_dict['batch_size']
            if not isinstance(bs, int):
                errors.append(f"batch_size: 必须是整数类型，当前类型为 {type(bs).__name__}")
            elif bs <= 0:
                errors.append(f"batch_size: 必须为正整数，当前值为 {bs}")
        
        # 验证num_episodes（训练回合数）
        if 'num_episodes' in config_dict:
            ne = config_dict['num_episodes']
            if not isinstance(ne, int):
                errors.append(f"num_episodes: 必须是整数类型，当前类型为 {type(ne).__name__}")
            elif ne <= 0:
                errors.append(f"num_episodes: 必须为正整数，当前值为 {ne}")
        
        # 验证discount_factor（折扣因子）
        if 'discount_factor' in config_dict:
            df = config_dict['discount_factor']
            if not isinstance(df, (int, float)):
                errors.append(f"discount_factor: 必须是数值类型，当前类型为 {type(df).__name__}")
            elif df < 0 or df > 1:
                errors.append(f"discount_factor: 必须在[0, 1]范围内，当前值为 {df}")
        
        # 验证network_architecture（网络架构）
        if 'network_architecture' in config_dict:
            na = config_dict['network_architecture']
            if not isinstance(na, list):
                errors.append(f"network_architecture: 必须是列表类型，当前类型为 {type(na).__name__}")
            elif len(na) == 0:
                errors.append("network_architecture: 不能为空列表")
            else:
                for i, dim in enumerate(na):
                    if not isinstance(dim, int):
                        errors.append(f"network_architecture[{i}]: 必须是整数，当前类型为 {type(dim).__name__}")
                    elif dim <= 0:
                        errors.append(f"network_architecture[{i}]: 必须为正整数，当前值为 {dim}")
        
        # 验证checkpoint_interval（检查点间隔）
        if 'checkpoint_interval' in config_dict:
            ci = config_dict['checkpoint_interval']
            if not isinstance(ci, int):
                errors.append(f"checkpoint_interval: 必须是整数类型，当前类型为 {type(ci).__name__}")
            elif ci <= 0:
                errors.append(f"checkpoint_interval: 必须为正整数，当前值为 {ci}")
        
        # 验证num_parallel_envs（并行环境数）
        if 'num_parallel_envs' in config_dict:
            npe = config_dict['num_parallel_envs']
            if not isinstance(npe, int):
                errors.append(f"num_parallel_envs: 必须是整数类型，当前类型为 {type(npe).__name__}")
            elif npe <= 0:
                errors.append(f"num_parallel_envs: 必须为正整数，当前值为 {npe}")
        
        # 验证initial_stack（初始筹码）
        if 'initial_stack' in config_dict:
            ist = config_dict['initial_stack']
            if not isinstance(ist, int):
                errors.append(f"initial_stack: 必须是整数类型，当前类型为 {type(ist).__name__}")
            elif ist <= 0:
                errors.append(f"initial_stack: 必须为正整数，当前值为 {ist}")
        
        # 验证small_blind（小盲注）
        if 'small_blind' in config_dict:
            sb = config_dict['small_blind']
            if not isinstance(sb, int):
                errors.append(f"small_blind: 必须是整数类型，当前类型为 {type(sb).__name__}")
            elif sb <= 0:
                errors.append(f"small_blind: 必须为正整数，当前值为 {sb}")
        
        # 验证big_blind（大盲注）
        if 'big_blind' in config_dict:
            bb = config_dict['big_blind']
            if not isinstance(bb, int):
                errors.append(f"big_blind: 必须是整数类型，当前类型为 {type(bb).__name__}")
            elif bb <= 0:
                errors.append(f"big_blind: 必须为正整数，当前值为 {bb}")
        
        # 验证盲注关系：大盲注必须大于小盲注
        if 'small_blind' in config_dict and 'big_blind' in config_dict:
            sb = config_dict['small_blind']
            bb = config_dict['big_blind']
            if isinstance(sb, int) and isinstance(bb, int) and sb > 0 and bb > 0:
                if bb <= sb:
                    errors.append(f"big_blind: 必须大于small_blind，当前big_blind={bb}, small_blind={sb}")
        
        # 验证初始筹码与盲注的关系
        if 'initial_stack' in config_dict and 'big_blind' in config_dict:
            ist = config_dict['initial_stack']
            bb = config_dict['big_blind']
            if isinstance(ist, int) and isinstance(bb, int) and ist > 0 and bb > 0:
                if ist < bb:
                    errors.append(f"initial_stack: 必须至少等于big_blind，当前initial_stack={ist}, big_blind={bb}")
        
        # 验证 entropy_coefficient（熵正则化系数）
        if 'entropy_coefficient' in config_dict:
            ec = config_dict['entropy_coefficient']
            if not isinstance(ec, (int, float)):
                errors.append(f"entropy_coefficient: 必须是数值类型，当前类型为 {type(ec).__name__}")
            elif ec < 0:
                errors.append(f"entropy_coefficient: 必须为非负数，当前值为 {ec}")
        
        # 验证 max_raises_per_street（每条街最大加注次数）
        if 'max_raises_per_street' in config_dict:
            mrps = config_dict['max_raises_per_street']
            if not isinstance(mrps, int):
                errors.append(f"max_raises_per_street: 必须是整数类型，当前类型为 {type(mrps).__name__}")
            elif mrps < 0:
                errors.append(f"max_raises_per_street: 必须为非负整数，当前值为 {mrps}")
        
        # 验证 regret_buffer_size（遗憾缓冲区大小）
        if 'regret_buffer_size' in config_dict:
            rbs = config_dict['regret_buffer_size']
            if not isinstance(rbs, int):
                errors.append(f"regret_buffer_size: 必须是整数类型，当前类型为 {type(rbs).__name__}")
            elif rbs <= 0:
                errors.append(f"regret_buffer_size: 必须为正整数，当前值为 {rbs}")
        
        # 验证 strategy_buffer_size（策略缓冲区大小）
        if 'strategy_buffer_size' in config_dict:
            sbs = config_dict['strategy_buffer_size']
            if not isinstance(sbs, int):
                errors.append(f"strategy_buffer_size: 必须是整数类型，当前类型为 {type(sbs).__name__}")
            elif sbs <= 0:
                errors.append(f"strategy_buffer_size: 必须为正整数，当前值为 {sbs}")
        
        # 验证 cfr_iterations_per_update（每次网络更新前的 CFR 迭代次数）
        if 'cfr_iterations_per_update' in config_dict:
            cipu = config_dict['cfr_iterations_per_update']
            if not isinstance(cipu, int):
                errors.append(f"cfr_iterations_per_update: 必须是整数类型，当前类型为 {type(cipu).__name__}")
            elif cipu <= 0:
                errors.append(f"cfr_iterations_per_update: 必须为正整数，当前值为 {cipu}")
        
        # 验证 network_train_steps（每次更新的训练步数）
        if 'network_train_steps' in config_dict:
            nts = config_dict['network_train_steps']
            if not isinstance(nts, int):
                errors.append(f"network_train_steps: 必须是整数类型，当前类型为 {type(nts).__name__}")
            elif nts <= 0:
                errors.append(f"network_train_steps: 必须为正整数，当前值为 {nts}")
        
        # 验证 use_abstraction（是否启用卡牌抽象）
        if 'use_abstraction' in config_dict:
            ua = config_dict['use_abstraction']
            if not isinstance(ua, bool):
                errors.append(f"use_abstraction: 必须是布尔类型，当前类型为 {type(ua).__name__}")
        
        # 验证 abstraction_path（抽象文件路径）
        if 'abstraction_path' in config_dict:
            ap = config_dict['abstraction_path']
            if not isinstance(ap, str):
                errors.append(f"abstraction_path: 必须是字符串类型，当前类型为 {type(ap).__name__}")
        
        # 验证 abstraction_config（抽象配置）
        if 'abstraction_config' in config_dict:
            ac = config_dict['abstraction_config']
            if not isinstance(ac, dict):
                errors.append(f"abstraction_config: 必须是字典类型，当前类型为 {type(ac).__name__}")
            else:
                # 验证抽象配置的具体参数
                abstraction_errors = self._validate_abstraction_config(ac)
                errors.extend(abstraction_errors)
        
        return errors
    
    def _validate_abstraction_config(self, config: Dict[str, Any]) -> List[str]:
        """验证抽象配置参数。
        
        Args:
            config: 抽象配置字典
            
        Returns:
            错误信息列表
        """
        errors = []
        
        # 验证桶数量参数
        bucket_params = ['preflop_buckets', 'flop_buckets', 'turn_buckets', 'river_buckets']
        for param in bucket_params:
            if param in config:
                value = config[param]
                if not isinstance(value, int):
                    errors.append(f"abstraction_config.{param}: 必须是整数类型，当前类型为 {type(value).__name__}")
                elif value <= 0:
                    errors.append(f"abstraction_config.{param}: 必须为正整数，当前值为 {value}")
        
        # 验证翻牌前桶数不超过169
        if 'preflop_buckets' in config:
            value = config['preflop_buckets']
            if isinstance(value, int) and value > 169:
                errors.append(f"abstraction_config.preflop_buckets: 不能超过169（起手牌类型数），当前值为 {value}")
        
        # 验证equity_bins参数
        if 'equity_bins' in config:
            value = config['equity_bins']
            if not isinstance(value, int):
                errors.append(f"abstraction_config.equity_bins: 必须是整数类型，当前类型为 {type(value).__name__}")
            elif value <= 0:
                errors.append(f"abstraction_config.equity_bins: 必须为正整数，当前值为 {value}")
        
        # 验证kmeans参数
        if 'kmeans_restarts' in config:
            value = config['kmeans_restarts']
            if not isinstance(value, int):
                errors.append(f"abstraction_config.kmeans_restarts: 必须是整数类型，当前类型为 {type(value).__name__}")
            elif value <= 0:
                errors.append(f"abstraction_config.kmeans_restarts: 必须为正整数，当前值为 {value}")
        
        if 'kmeans_max_iters' in config:
            value = config['kmeans_max_iters']
            if not isinstance(value, int):
                errors.append(f"abstraction_config.kmeans_max_iters: 必须是整数类型，当前类型为 {type(value).__name__}")
            elif value <= 0:
                errors.append(f"abstraction_config.kmeans_max_iters: 必须为正整数，当前值为 {value}")
        
        # 验证布尔参数
        if 'use_potential_aware' in config:
            value = config['use_potential_aware']
            if not isinstance(value, bool):
                errors.append(f"abstraction_config.use_potential_aware: 必须是布尔类型，当前类型为 {type(value).__name__}")
        
        # 验证随机种子
        if 'random_seed' in config:
            value = config['random_seed']
            if not isinstance(value, int):
                errors.append(f"abstraction_config.random_seed: 必须是整数类型，当前类型为 {type(value).__name__}")
        
        # 验证工作进程数
        if 'num_workers' in config:
            value = config['num_workers']
            if not isinstance(value, int):
                errors.append(f"abstraction_config.num_workers: 必须是整数类型，当前类型为 {type(value).__name__}")
            elif value < 0:
                errors.append(f"abstraction_config.num_workers: 不能为负数，当前值为 {value}")
        
        return errors
    
    def _apply_defaults(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """为缺失的可选参数应用默认值，并处理废弃参数。
        
        Args:
            config_dict: 原始配置字典
            
        Returns:
            Dict[str, Any]: 应用默认值后的配置字典
        """
        result = dict(config_dict)
        
        # 移除废弃的参数（如 cfr_weight）以保持向后兼容
        for deprecated_param in DEPRECATED_PARAMS:
            if deprecated_param in result:
                del result[deprecated_param]
        
        # 应用默认值到缺失的参数
        for param, default_value in DEFAULT_CONFIG.items():
            if param not in result:
                result[param] = default_value
        
        return result
    
    def get_default_config(self) -> TrainingConfig:
        """获取默认配置。
        
        Returns:
            TrainingConfig: 使用所有默认值的配置对象
        """
        return TrainingConfig(**DEFAULT_CONFIG)
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并两个配置字典，override中的值会覆盖base中的值。
        
        Args:
            base: 基础配置字典
            override: 覆盖配置字典
            
        Returns:
            Dict[str, Any]: 合并后的配置字典
        """
        result = dict(base)
        result.update(override)
        return result
