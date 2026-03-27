"""检查点管理器模块 - 处理模型检查点的保存、加载和管理。

本模块实现了训练过程中模型状态的持久化管理：
- 保存模型参数、优化器状态和元数据
- 从检查点文件加载模型状态
- 列出所有可用检查点
- 删除指定检查点

支持两种检查点格式：
- Deep CFR格式（v1）：包含regret_network和policy_network
- 旧格式（legacy）：包含policy_network和value_network
"""

import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from models.core import CheckpointInfo

# 检查点格式版本
CHECKPOINT_FORMAT_VERSION = 'deep_cfr_v1'


class CheckpointManager:
    """检查点管理器 - 负责模型检查点的保存、加载和管理。
    
    提供以下功能：
    - 保存模型、优化器状态和元数据到.pt文件
    - 从.pt文件加载检查点
    - 列出所有检查点及其信息
    - 删除指定检查点文件
    
    文件命名格式：checkpoint_{timestamp}_{episode}.pt
    
    Attributes:
        checkpoint_dir: 检查点保存目录
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path] = "checkpoints"):
        """初始化检查点管理器。
        
        Args:
            checkpoint_dir: 检查点保存目录，默认为"checkpoints"
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        # 确保目录存在
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        metadata: Dict[str, Any],
        value_network: Optional[nn.Module] = None,
        value_optimizer: Optional[Optimizer] = None
    ) -> str:
        """保存检查点（旧格式，保持向后兼容）。
        
        将模型参数、优化器状态和元数据保存到.pt文件。
        支持同时保存策略网络和价值网络。
        文件名格式：checkpoint_{timestamp}_{episode}.pt
        
        注意：推荐使用 save_deep_cfr() 方法保存 Deep CFR 格式的检查点。
        
        Args:
            model: 要保存的策略网络
            optimizer: 策略网络优化器（可选）
            metadata: 元数据字典，应包含：
                - episode_number: 训练回合数
                - win_rate: 胜率（可选，默认0.0）
                - avg_reward: 平均奖励（可选，默认0.0）
            value_network: 价值网络（可选）
            value_optimizer: 价值网络优化器（可选）
                
        Returns:
            str: 保存的检查点文件路径
            
        Raises:
            ValueError: 元数据缺少必需字段
            IOError: 文件保存失败
        """
        # 验证必需的元数据字段
        if 'episode_number' not in metadata:
            raise ValueError("元数据必须包含'episode_number'字段")
        
        episode_number = metadata['episode_number']
        
        # 生成唯一文件名
        # 使用时间戳（精确到微秒）确保唯一性
        timestamp = int(time.time() * 1000000)
        filename = f"checkpoint_{timestamp}_{episode_number}.pt"
        filepath = self.checkpoint_dir / filename
        
        # 准备保存的数据
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'episode_number': episode_number,
            'timestamp': datetime.now().isoformat(),
            'win_rate': metadata.get('win_rate', 0.0),
            'avg_reward': metadata.get('avg_reward', 0.0),
        }
        
        # 保存价值网络（如果提供）
        if value_network is not None:
            checkpoint_data['value_network_state_dict'] = value_network.state_dict()
            checkpoint_data['has_value_network'] = True
        else:
            checkpoint_data['has_value_network'] = False
        
        if value_optimizer is not None:
            checkpoint_data['value_optimizer_state_dict'] = value_optimizer.state_dict()
        
        # 添加其他元数据
        for key, value in metadata.items():
            if key not in checkpoint_data:
                checkpoint_data[key] = value
        
        # 保存检查点
        torch.save(checkpoint_data, filepath)
        
        return str(filepath)
    
    def save_deep_cfr(
        self,
        regret_network: nn.Module,
        policy_network: nn.Module,
        regret_optimizer: Optional[Optimizer],
        policy_optimizer: Optional[Optimizer],
        metadata: Dict[str, Any]
    ) -> str:
        """保存 Deep CFR 格式的检查点。
        
        将遗憾网络、策略网络及其优化器状态保存到.pt文件。
        文件名格式：checkpoint_{timestamp}_{episode}.pt
        
        Args:
            regret_network: 遗憾网络
            policy_network: 策略网络
            regret_optimizer: 遗憾网络优化器（可选）
            policy_optimizer: 策略网络优化器（可选）
            metadata: 元数据字典，应包含：
                - episode_number: 训练回合数（CFR迭代次数）
                - win_rate: 胜率（可选，默认0.0）
                - avg_reward: 平均奖励（可选，默认0.0）
                
        Returns:
            str: 保存的检查点文件路径
            
        Raises:
            ValueError: 元数据缺少必需字段
            IOError: 文件保存失败
        """
        # 验证必需的元数据字段
        if 'episode_number' not in metadata:
            raise ValueError("元数据必须包含'episode_number'字段")
        
        episode_number = metadata['episode_number']
        
        # 生成唯一文件名
        timestamp = int(time.time() * 1000000)
        filename = f"checkpoint_{timestamp}_{episode_number}.pt"
        filepath = self.checkpoint_dir / filename
        
        # 准备保存的数据（Deep CFR格式）
        checkpoint_data = {
            # 格式版本号
            'checkpoint_format': CHECKPOINT_FORMAT_VERSION,
            # 遗憾网络参数
            'regret_network_state_dict': regret_network.state_dict(),
            # 策略网络参数
            'policy_network_state_dict': policy_network.state_dict(),
            # 遗憾网络优化器参数
            'regret_optimizer_state_dict': regret_optimizer.state_dict() if regret_optimizer else None,
            # 策略网络优化器参数
            'policy_optimizer_state_dict': policy_optimizer.state_dict() if policy_optimizer else None,
            # 元数据
            'episode_number': episode_number,
            'timestamp': datetime.now().isoformat(),
            'win_rate': metadata.get('win_rate', 0.0),
            'avg_reward': metadata.get('avg_reward', 0.0),
        }
        
        # 添加其他元数据
        for key, value in metadata.items():
            if key not in checkpoint_data:
                checkpoint_data[key] = value
        
        # 保存检查点
        torch.save(checkpoint_data, filepath)
        
        return str(filepath)
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        value_network: Optional[nn.Module] = None,
        value_optimizer: Optional[Optimizer] = None
    ) -> Tuple[nn.Module, Optional[Optimizer], Dict[str, Any]]:
        """加载检查点（旧格式，保持向后兼容）。
        
        从.pt文件加载模型参数、优化器状态和元数据。
        支持同时加载策略网络和价值网络。
        
        注意：推荐使用 load_deep_cfr() 方法加载 Deep CFR 格式的检查点。
        
        Args:
            checkpoint_path: 检查点文件路径
            model: 要加载参数的策略网络（必须与保存时的架构匹配）
            optimizer: 策略网络优化器（可选）
            value_network: 价值网络（可选）
            value_optimizer: 价值网络优化器（可选）
            
        Returns:
            Tuple[nn.Module, Optional[Optimizer], Dict[str, Any]]:
                - 加载参数后的策略网络
                - 加载状态后的优化器（如果提供）
                - 元数据字典
                
        Raises:
            FileNotFoundError: 检查点文件不存在
            RuntimeError: 模型架构不匹配或文件损坏
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 加载检查点数据
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        
        # 加载策略网络参数
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # 加载策略网络优化器状态（如果提供了优化器且检查点中有优化器状态）
        if optimizer is not None and checkpoint_data.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # 加载价值网络参数（如果提供了价值网络且检查点中有价值网络状态）
        if value_network is not None and checkpoint_data.get('value_network_state_dict') is not None:
            value_network.load_state_dict(checkpoint_data['value_network_state_dict'])
        
        # 加载价值网络优化器状态
        if value_optimizer is not None and checkpoint_data.get('value_optimizer_state_dict') is not None:
            value_optimizer.load_state_dict(checkpoint_data['value_optimizer_state_dict'])
        
        # 提取元数据
        metadata = {
            'episode_number': checkpoint_data.get('episode_number', 0),
            'timestamp': checkpoint_data.get('timestamp', ''),
            'win_rate': checkpoint_data.get('win_rate', 0.0),
            'avg_reward': checkpoint_data.get('avg_reward', 0.0),
            'has_value_network': checkpoint_data.get('has_value_network', False),
        }
        
        # 添加其他元数据
        excluded_keys = {'model_state_dict', 'optimizer_state_dict', 
                        'value_network_state_dict', 'value_optimizer_state_dict',
                        'episode_number', 'timestamp', 'win_rate', 'avg_reward',
                        'has_value_network'}
        for key, value in checkpoint_data.items():
            if key not in excluded_keys:
                metadata[key] = value
        
        return model, optimizer, metadata
    
    def load_deep_cfr(
        self,
        checkpoint_path: Union[str, Path],
        regret_network: nn.Module,
        policy_network: nn.Module,
        regret_optimizer: Optional[Optimizer] = None,
        policy_optimizer: Optional[Optimizer] = None
    ) -> Tuple[nn.Module, nn.Module, Optional[Optimizer], Optional[Optimizer], Dict[str, Any]]:
        """加载 Deep CFR 格式的检查点。
        
        支持两种检查点格式：
        1. Deep CFR格式（v1）：包含regret_network和policy_network
        2. 旧格式（legacy）：包含policy_network和value_network（兼容性处理）
        
        Args:
            checkpoint_path: 检查点文件路径
            regret_network: 遗憾网络（必须与保存时的架构匹配）
            policy_network: 策略网络（必须与保存时的架构匹配）
            regret_optimizer: 遗憾网络优化器（可选）
            policy_optimizer: 策略网络优化器（可选）
            
        Returns:
            Tuple[nn.Module, nn.Module, Optional[Optimizer], Optional[Optimizer], Dict[str, Any]]:
                - 加载参数后的遗憾网络
                - 加载参数后的策略网络
                - 加载状态后的遗憾网络优化器（如果提供）
                - 加载状态后的策略网络优化器（如果提供）
                - 元数据字典
                
        Raises:
            FileNotFoundError: 检查点文件不存在
            RuntimeError: 模型架构不匹配或文件损坏
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 加载检查点数据
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        
        # 检测检查点格式
        checkpoint_format = checkpoint_data.get('checkpoint_format', 'legacy')
        
        if checkpoint_format == CHECKPOINT_FORMAT_VERSION:
            # Deep CFR格式
            self._load_deep_cfr_format(
                checkpoint_data, regret_network, policy_network,
                regret_optimizer, policy_optimizer
            )
        else:
            # 旧格式：兼容性处理
            self._load_legacy_format_as_deep_cfr(
                checkpoint_data, regret_network, policy_network,
                regret_optimizer, policy_optimizer
            )
        
        # 提取元数据
        metadata = self._extract_metadata(checkpoint_data)
        metadata['checkpoint_format'] = checkpoint_format
        
        return regret_network, policy_network, regret_optimizer, policy_optimizer, metadata
    
    def _load_deep_cfr_format(
        self,
        checkpoint_data: Dict[str, Any],
        regret_network: nn.Module,
        policy_network: nn.Module,
        regret_optimizer: Optional[Optimizer],
        policy_optimizer: Optional[Optimizer]
    ) -> None:
        """加载 Deep CFR 格式的检查点数据。
        
        Args:
            checkpoint_data: 检查点数据字典
            regret_network: 遗憾网络
            policy_network: 策略网络
            regret_optimizer: 遗憾网络优化器
            policy_optimizer: 策略网络优化器
        """
        # 加载遗憾网络参数
        if 'regret_network_state_dict' in checkpoint_data:
            regret_network.load_state_dict(checkpoint_data['regret_network_state_dict'])
        
        # 加载策略网络参数
        if 'policy_network_state_dict' in checkpoint_data:
            policy_network.load_state_dict(checkpoint_data['policy_network_state_dict'])
        
        # 加载遗憾网络优化器状态
        if regret_optimizer is not None and checkpoint_data.get('regret_optimizer_state_dict') is not None:
            regret_optimizer.load_state_dict(checkpoint_data['regret_optimizer_state_dict'])
        
        # 加载策略网络优化器状态
        if policy_optimizer is not None and checkpoint_data.get('policy_optimizer_state_dict') is not None:
            policy_optimizer.load_state_dict(checkpoint_data['policy_optimizer_state_dict'])
    
    def _load_legacy_format_as_deep_cfr(
        self,
        checkpoint_data: Dict[str, Any],
        regret_network: nn.Module,
        policy_network: nn.Module,
        regret_optimizer: Optional[Optimizer],
        policy_optimizer: Optional[Optimizer]
    ) -> None:
        """加载旧格式检查点并转换为 Deep CFR 格式。
        
        旧格式包含 policy_network 和 value_network。
        我们只加载 policy_network，遗憾网络保持初始化状态。
        
        Args:
            checkpoint_data: 检查点数据字典
            regret_network: 遗憾网络（保持初始化状态）
            policy_network: 策略网络
            regret_optimizer: 遗憾网络优化器（不加载）
            policy_optimizer: 策略网络优化器
        """
        warnings.warn(
            "检测到旧格式检查点，仅加载策略网络参数。"
            "遗憾网络将保持初始化状态。"
            "建议使用新格式重新保存检查点。",
            UserWarning
        )
        
        # 加载策略网络（旧格式中的 model_state_dict）
        if 'model_state_dict' in checkpoint_data:
            policy_network.load_state_dict(checkpoint_data['model_state_dict'])
        
        # 加载策略网络优化器
        if policy_optimizer is not None and checkpoint_data.get('optimizer_state_dict') is not None:
            policy_optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # 旧格式的 value_network 不再使用，跳过
        if checkpoint_data.get('has_value_network'):
            warnings.warn(
                "旧检查点中的价值网络参数已被忽略（Deep CFR 不使用价值网络）",
                UserWarning
            )
    
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
        }
        
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
            if key not in excluded_keys:
                metadata[key] = value
        
        return metadata
    
    def detect_checkpoint_format(self, checkpoint_path: Union[str, Path]) -> str:
        """检测检查点文件的格式。
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            检查点格式字符串：'deep_cfr_v1' 或 'legacy'
            
        Raises:
            FileNotFoundError: 检查点文件不存在
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        return checkpoint_data.get('checkpoint_format', 'legacy')
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """列出所有可用检查点。
        
        扫描检查点目录，返回所有检查点的信息。
        
        Returns:
            List[CheckpointInfo]: 检查点信息列表，按时间戳排序（最新的在前）
        """
        checkpoints = []
        
        # 扫描目录中的.pt文件
        for filepath in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                # 加载检查点元数据（不加载模型参数以提高效率）
                checkpoint_data = torch.load(filepath, weights_only=False)
                
                # 解析时间戳
                timestamp_str = checkpoint_data.get('timestamp', '')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    except ValueError:
                        timestamp = datetime.now()
                else:
                    # 从文件修改时间获取
                    timestamp = datetime.fromtimestamp(filepath.stat().st_mtime)
                
                # 创建CheckpointInfo对象
                info = CheckpointInfo(
                    path=str(filepath),
                    episode_number=checkpoint_data.get('episode_number', 0),
                    timestamp=timestamp,
                    win_rate=checkpoint_data.get('win_rate', 0.0),
                    avg_reward=checkpoint_data.get('avg_reward', 0.0)
                )
                checkpoints.append(info)
                
            except Exception as e:
                # 跳过无法读取的文件
                print(f"警告：无法读取检查点文件 {filepath}: {e}")
                continue
        
        # 按时间戳排序（最新的在前）
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        
        return checkpoints
    
    def delete(self, checkpoint_path: Union[str, Path]) -> bool:
        """删除指定检查点。
        
        Args:
            checkpoint_path: 要删除的检查点文件路径
            
        Returns:
            bool: 删除成功返回True，文件不存在返回False
            
        Raises:
            IOError: 删除操作失败
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return False
        
        # 删除文件
        checkpoint_path.unlink()
        return True
    
    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """获取最新的检查点。
        
        Returns:
            Optional[CheckpointInfo]: 最新检查点的信息，如果没有检查点则返回None
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None
    
    def get_checkpoint_by_episode(self, episode_number: int) -> Optional[CheckpointInfo]:
        """根据回合数获取检查点。
        
        Args:
            episode_number: 目标回合数
            
        Returns:
            Optional[CheckpointInfo]: 匹配的检查点信息，如果没有找到则返回None
        """
        checkpoints = self.list_checkpoints()
        for checkpoint in checkpoints:
            if checkpoint.episode_number == episode_number:
                return checkpoint
        return None
    
    def cleanup_old_checkpoints(self, keep_count: int = 5) -> int:
        """清理旧检查点，只保留最新的N个。
        
        Args:
            keep_count: 要保留的检查点数量
            
        Returns:
            int: 删除的检查点数量
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_count:
            return 0
        
        # 删除多余的检查点
        deleted_count = 0
        for checkpoint in checkpoints[keep_count:]:
            if self.delete(checkpoint.path):
                deleted_count += 1
        
        return deleted_count
