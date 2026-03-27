"""模型加载器模块 - 封装检查点加载逻辑，处理各种错误情况。

本模块实现了策略查看器的模型加载功能：
- 加载训练好的模型检查点
- 验证检查点文件的有效性
- 提取模型元数据信息
- 自动检测动作空间维度和配置
- 处理各种加载错误情况
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch

from models.networks import PolicyNetwork, ValueNetwork, RegretNetwork
from utils.checkpoint_manager import CheckpointManager
from utils.exceptions import (
    CheckpointNotFoundError,
    CheckpointCorruptedError,
    ModelLoadError,
)
from viewer.models import ActionConfig


@dataclass
class ModelMetadata:
    """模型元数据信息。
    
    Attributes:
        checkpoint_path: 检查点文件路径
        episode_number: 训练迭代次数
        timestamp: 保存时间戳
        win_rate: 胜率
        avg_reward: 平均奖励
        extra_info: 其他额外信息
    """
    checkpoint_path: str
    episode_number: int
    timestamp: str
    win_rate: float
    avg_reward: float
    extra_info: Dict[str, Any]
    
    def get_formatted_timestamp(self) -> str:
        """获取格式化的时间戳字符串。"""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return self.timestamp or "未知"
    
    def get_summary(self) -> str:
        """获取元数据摘要。"""
        return (
            f"检查点: {Path(self.checkpoint_path).name}\n"
            f"训练迭代: {self.episode_number}\n"
            f"保存时间: {self.get_formatted_timestamp()}\n"
            f"胜率: {self.win_rate:.2%}\n"
            f"平均奖励: {self.avg_reward:.4f}"
        )


class ModelLoader:
    """模型加载器 - 负责加载和管理训练好的模型。
    
    提供以下功能：
    - 加载检查点文件
    - 验证检查点有效性
    - 提取模型元数据
    - 错误处理和状态管理
    
    支持两种检查点格式：
    - Deep CFR格式（新）：包含regret_network和policy_network
    - 旧格式：包含policy_network和value_network
    
    Attributes:
        device: 计算设备（cpu或cuda）
        policy_network: 加载的策略网络
        value_network: 加载的价值网络（旧格式，可选）
        regret_network: 加载的遗憾网络（Deep CFR格式，可选）
        metadata: 模型元数据
    """
    
    def __init__(self, device: str = "cpu"):
        """初始化模型加载器。
        
        Args:
            device: 计算设备，默认为"cpu"
        """
        self.device = device
        self._policy_network: Optional[PolicyNetwork] = None
        self._value_network: Optional[ValueNetwork] = None
        self._regret_network: Optional[RegretNetwork] = None
        self._metadata: Optional[ModelMetadata] = None
        self._checkpoint_manager: Optional[CheckpointManager] = None
        self._checkpoint_format: str = "unknown"  # "deep_cfr_v1" 或 "legacy"
        self._action_config: Optional[ActionConfig] = None
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        input_dim: int = 370,
        hidden_dims: Optional[List[int]] = None,
        action_dim: int = 6
    ) -> ModelMetadata:
        """加载模型检查点。
        
        支持两种检查点格式：
        - Deep CFR格式（新）：包含regret_network和policy_network
        - 旧格式：包含policy_network和value_network
        
        Args:
            checkpoint_path: 检查点文件路径
            input_dim: 输入维度，默认370
            hidden_dims: 隐藏层维度列表，默认[512, 256, 128]
            action_dim: 行动空间维度，默认5（FOLD, CHECK, CALL, RAISE_SMALL, RAISE_BIG）
            
        Returns:
            ModelMetadata: 模型元数据信息
            
        Raises:
            CheckpointNotFoundError: 检查点文件不存在
            CheckpointCorruptedError: 检查点文件损坏或格式无效
            ModelLoadError: 模型加载失败（如架构不匹配）
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        checkpoint_path = Path(checkpoint_path)
        
        # 验证文件存在性
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(str(checkpoint_path))
        
        # 验证文件扩展名
        if checkpoint_path.suffix not in ['.pt', '.pth']:
            raise CheckpointCorruptedError(
                str(checkpoint_path),
                f"不支持的文件格式: {checkpoint_path.suffix}，期望 .pt 或 .pth"
            )
        
        # 尝试加载检查点数据
        try:
            checkpoint_data = torch.load(
                checkpoint_path, 
                map_location=self.device,
                weights_only=False
            )
        except Exception as e:
            raise CheckpointCorruptedError(
                str(checkpoint_path),
                f"无法读取检查点文件: {str(e)}"
            )
        
        # 验证检查点数据结构
        if not isinstance(checkpoint_data, dict):
            raise CheckpointCorruptedError(
                str(checkpoint_path),
                "检查点数据格式无效：期望字典类型"
            )
        
        # 检测检查点格式
        self._checkpoint_format = checkpoint_data.get('checkpoint_format', 'legacy')
        
        # 检测动作配置
        detected_action_dim = self._detect_action_dim(checkpoint_data)
        self._action_config = self._create_action_config(checkpoint_data, detected_action_dim)
        
        # 使用检测到的动作维度（如果与传入参数不同，优先使用检测值）
        actual_action_dim = self._action_config.action_dim
        
        if self._checkpoint_format == 'deep_cfr_v1':
            # Deep CFR 格式
            self._load_deep_cfr_checkpoint(checkpoint_data, input_dim, hidden_dims, actual_action_dim)
        else:
            # 旧格式
            self._load_legacy_checkpoint(checkpoint_data, input_dim, hidden_dims, actual_action_dim, checkpoint_path)
        
        # 提取元数据
        self._metadata = ModelMetadata(
            checkpoint_path=str(checkpoint_path),
            episode_number=checkpoint_data.get('episode_number', 0),
            timestamp=checkpoint_data.get('timestamp', ''),
            win_rate=checkpoint_data.get('win_rate', 0.0),
            avg_reward=checkpoint_data.get('avg_reward', 0.0),
            extra_info={
                k: v for k, v in checkpoint_data.items()
                if k not in ['model_state_dict', 'optimizer_state_dict',
                            'value_network_state_dict', 'value_optimizer_state_dict',
                            'regret_network_state_dict', 'policy_network_state_dict',
                            'regret_optimizer_state_dict', 'policy_optimizer_state_dict',
                            'episode_number', 'timestamp', 'win_rate', 'avg_reward',
                            'has_value_network', 'checkpoint_format']
            }
        )
        
        # 添加网络状态到元数据
        self._metadata.extra_info['has_value_network'] = self._value_network is not None
        self._metadata.extra_info['has_regret_network'] = self._regret_network is not None
        self._metadata.extra_info['checkpoint_format'] = self._checkpoint_format
        self._metadata.extra_info['action_dim'] = self._action_config.action_dim
        self._metadata.extra_info['action_names'] = self._action_config.action_names
        
        return self._metadata
    
    def _load_deep_cfr_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        input_dim: int,
        hidden_dims: List[int],
        action_dim: int
    ) -> None:
        """加载 Deep CFR 格式的检查点。
        
        Args:
            checkpoint_data: 检查点数据字典
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            action_dim: 行动空间维度
        """
        # 加载遗憾网络
        if 'regret_network_state_dict' in checkpoint_data:
            try:
                self._regret_network = RegretNetwork(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    action_dim=action_dim
                ).to(self.device)
                self._regret_network.load_state_dict(checkpoint_data['regret_network_state_dict'])
                self._regret_network.eval()
            except Exception as e:
                print(f"警告：遗憾网络加载失败: {e}")
                self._regret_network = None
        
        # 加载策略网络
        if 'policy_network_state_dict' in checkpoint_data:
            try:
                self._policy_network = PolicyNetwork(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    action_dim=action_dim
                ).to(self.device)
                self._policy_network.load_state_dict(checkpoint_data['policy_network_state_dict'])
                self._policy_network.eval()
            except Exception as e:
                print(f"警告：策略网络加载失败: {e}")
                self._policy_network = None
        
        # Deep CFR 格式没有价值网络
        self._value_network = None
    
    def _load_legacy_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        input_dim: int,
        hidden_dims: List[int],
        action_dim: int,
        checkpoint_path: Path
    ) -> None:
        """加载旧格式的检查点。
        
        Args:
            checkpoint_data: 检查点数据字典
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            action_dim: 行动空间维度
            checkpoint_path: 检查点路径（用于错误信息）
        """
        if 'model_state_dict' not in checkpoint_data:
            raise CheckpointCorruptedError(
                str(checkpoint_path),
                "检查点缺少必需的 'model_state_dict' 字段"
            )
        
        # 创建策略网络
        try:
            self._policy_network = PolicyNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                action_dim=action_dim
            ).to(self.device)
        except Exception as e:
            raise ModelLoadError(
                str(checkpoint_path),
                f"创建策略网络失败: {str(e)}"
            )
        
        # 加载模型参数
        try:
            self._policy_network.load_state_dict(checkpoint_data['model_state_dict'])
        except Exception as e:
            self._policy_network = None
            raise ModelLoadError(
                str(checkpoint_path),
                f"模型参数加载失败（可能是架构不匹配）: {str(e)}"
            )
        
        self._policy_network.eval()
        
        # 尝试加载价值网络（如果检查点中包含）
        if checkpoint_data.get('has_value_network') or checkpoint_data.get('value_network_state_dict'):
            try:
                self._value_network = ValueNetwork(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims
                ).to(self.device)
                self._value_network.load_state_dict(checkpoint_data['value_network_state_dict'])
                self._value_network.eval()
            except Exception as e:
                print(f"警告：价值网络加载失败: {e}")
                self._value_network = None
        
        # 旧格式没有遗憾网络
        self._regret_network = None
    
    def _detect_action_dim(self, checkpoint_data: Dict[str, Any]) -> int:
        """从检查点数据中检测动作空间维度。
        
        检测顺序：
        1. 检查 action_config 元数据中的 action_dim
        2. 检查顶层的 action_dim 字段
        3. 从策略网络权重检测输出层维度
        4. 从遗憾网络权重检测输出层维度
        5. 从旧格式的 model_state_dict 检测
        6. 默认返回 6
        
        Args:
            checkpoint_data: 检查点数据字典
            
        Returns:
            检测到的动作维度
        """
        # 1. 检查 action_config 元数据
        if 'action_config' in checkpoint_data:
            config = checkpoint_data['action_config']
            if 'action_dim' in config:
                return config['action_dim']
            if 'action_names' in config:
                return len(config['action_names'])
        
        # 2. 检查顶层的 action_dim 字段
        if 'action_dim' in checkpoint_data:
            return checkpoint_data['action_dim']
        
        # 3. 从策略网络权重检测
        if 'policy_network_state_dict' in checkpoint_data:
            dim = self._detect_dim_from_state_dict(
                checkpoint_data['policy_network_state_dict']
            )
            if dim is not None:
                return dim
        
        # 4. 从遗憾网络权重检测
        if 'regret_network_state_dict' in checkpoint_data:
            dim = self._detect_dim_from_state_dict(
                checkpoint_data['regret_network_state_dict']
            )
            if dim is not None:
                return dim
        
        # 5. 从旧格式的 model_state_dict 检测
        if 'model_state_dict' in checkpoint_data:
            dim = self._detect_dim_from_state_dict(
                checkpoint_data['model_state_dict']
            )
            if dim is not None:
                return dim
        
        # 6. 默认返回 6
        return 6
    
    def _detect_dim_from_state_dict(
        self, 
        state_dict: Dict[str, torch.Tensor]
    ) -> Optional[int]:
        """从网络状态字典中检测输出层维度。
        
        查找最后一个线性层的权重，其输出维度即为动作维度。
        网络结构为 nn.Sequential，键名格式为 'network.N.weight'。
        
        Args:
            state_dict: 网络状态字典
            
        Returns:
            检测到的维度，如果无法检测则返�� None
        """
        # 查找所有权重键
        weight_keys = [k for k in state_dict.keys() if k.endswith('.weight')]
        
        if not weight_keys:
            return None
        
        # 按键名排序，找到最后一个权重层
        # 键名格式: 'network.0.weight', 'network.2.weight', ...
        # 最后一个数字最大的是输出层
        def get_layer_index(key: str) -> int:
            """从键名中提取层索引。"""
            parts = key.split('.')
            for part in parts:
                if part.isdigit():
                    return int(part)
            return -1
        
        # 按层索引排序
        sorted_keys = sorted(weight_keys, key=get_layer_index)
        last_weight_key = sorted_keys[-1]
        
        # 获取输出维度（权重矩阵的第一个维度）
        weight = state_dict[last_weight_key]
        output_dim = weight.shape[0]
        
        # 验证维度是否合理（动作维度通常在 2-10 之间）
        if 2 <= output_dim <= 10:
            return output_dim
        
        return None
    
    def _create_action_config(
        self, 
        checkpoint_data: Dict[str, Any],
        detected_dim: int
    ) -> ActionConfig:
        """创建动作配置。
        
        如果检查点包含 action_config 元数据，则使用该配置；
        否则根据检测到的维度使用默认映射。
        
        Args:
            checkpoint_data: 检查点数据字典
            detected_dim: 检测到的动作维度
            
        Returns:
            ActionConfig 实例
        """
        # 检查是否有 action_config 元数据
        if 'action_config' in checkpoint_data:
            try:
                return ActionConfig.from_checkpoint(checkpoint_data)
            except (ValueError, KeyError) as e:
                # 如果元数据���效，使用默认配置
                print(f"警告：action_config 元数据无效 ({e})，使用默认配置")
        
        # 使用默认配置
        return ActionConfig.default_for_dim(detected_dim)
    
    def validate_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """验证检查点文件是否有效。
        
        不加载模型，仅验证文件格式和必需字段。
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            bool: 检查点有效返回True，否则返回False
        """
        checkpoint_path = Path(checkpoint_path)
        
        # 检查文件存在性
        if not checkpoint_path.exists():
            return False
        
        # 检查文件扩展名
        if checkpoint_path.suffix not in ['.pt', '.pth']:
            return False
        
        # 尝试加载并验证结构
        try:
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False
            )
            
            # 验证是字典类型
            if not isinstance(checkpoint_data, dict):
                return False
            
            # 验证必需字段
            if 'model_state_dict' not in checkpoint_data:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_checkpoint_info(
        self, 
        checkpoint_path: Union[str, Path]
    ) -> Optional[ModelMetadata]:
        """获取检查点的元数据信息（不加载模型）。
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            ModelMetadata: 元数据信息，如果文件无效则返回None
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not self.validate_checkpoint(checkpoint_path):
            return None
        
        try:
            checkpoint_data = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False
            )
            
            return ModelMetadata(
                checkpoint_path=str(checkpoint_path),
                episode_number=checkpoint_data.get('episode_number', 0),
                timestamp=checkpoint_data.get('timestamp', ''),
                win_rate=checkpoint_data.get('win_rate', 0.0),
                avg_reward=checkpoint_data.get('avg_reward', 0.0),
                extra_info={
                    k: v for k, v in checkpoint_data.items()
                    if k not in ['model_state_dict', 'optimizer_state_dict',
                                'episode_number', 'timestamp', 'win_rate', 'avg_reward']
                }
            )
        except Exception:
            return None
    
    def unload(self) -> None:
        """卸载当前加载的模型，释放资源。"""
        self._policy_network = None
        self._value_network = None
        self._regret_network = None
        self._metadata = None
        self._checkpoint_format = "unknown"
        self._action_config = None
    
    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self._policy_network is not None
    
    @property
    def policy_network(self) -> Optional[PolicyNetwork]:
        """获取加载的策略网络。"""
        return self._policy_network
    
    @property
    def value_network(self) -> Optional[ValueNetwork]:
        """获取加载的价值网络。"""
        return self._value_network
    
    @property
    def has_value_network(self) -> bool:
        """检查是否有价值网络可用。"""
        return self._value_network is not None
    
    @property
    def regret_network(self) -> Optional[RegretNetwork]:
        """获取加载的遗憾网络。"""
        return self._regret_network
    
    @property
    def has_regret_network(self) -> bool:
        """检查是否有遗憾网络可用。"""
        return self._regret_network is not None
    
    @property
    def checkpoint_format(self) -> str:
        """获取检查点格式（'deep_cfr_v1' 或 'legacy'）。"""
        return self._checkpoint_format
    
    @property
    def metadata(self) -> Optional[ModelMetadata]:
        """获取模型元数据。"""
        return self._metadata
    
    @property
    def action_config(self) -> Optional[ActionConfig]:
        """获取动作配置。
        
        Returns:
            ActionConfig 实例，如果模型未加载则返回 None
        """
        return self._action_config

