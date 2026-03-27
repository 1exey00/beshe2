"""错误恢复策略模块 - 提供错误恢复和重试机制。

该模块实现以下错误恢复策略：
1. 检查点恢复 - 训练中断时从最近的检查点恢复
2. 自动重试 - 对于临时性错误自动重试有限次数
3. 降级运行 - GPU不可用时自动切换到CPU训练
4. 数据验证 - 加载检查点前验证文件完整性
"""

import functools
import time
import os
import hashlib
from typing import Callable, TypeVar, Optional, Any, List, Type, Tuple
from pathlib import Path

from utils.logger import get_logger
from utils.exceptions import (
    PokerAIError,
    CheckpointCorruptedError,
    CheckpointNotFoundError,
    DiskSpaceError,
    GPUError,
    TrainingInterruptedError,
    ResourceError
)


logger = get_logger('error_recovery')

# 类型变量用于泛型函数
T = TypeVar('T')


class RetryConfig:
    """重试配置类。
    
    Attributes:
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_backoff: 是否使用指数退避
        retry_exceptions: 需要重试的异常类型列表
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        retry_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """初始化重试配置。
        
        Args:
            max_retries: 最大重试次数
            initial_delay: 初始延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            exponential_backoff: 是否使用指数退避
            retry_exceptions: 需要重试的异常类型列表
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.retry_exceptions = retry_exceptions or [
            IOError,
            OSError,
            ConnectionError,
            TimeoutError
        ]


# 默认重试配置
DEFAULT_RETRY_CONFIG = RetryConfig()


def retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """重试装饰器。
    
    对于临时性错误（如网络IO），自动重试有限次数。
    
    Args:
        config: 重试配置，如果为None则使用默认配置
        on_retry: 重试时的回调函数，接收异常和重试次数
        
    Returns:
        装饰器函数
        
    Example:
        @retry(RetryConfig(max_retries=5))
        def fetch_data():
            # 可能失败的操作
            pass
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = config.initial_delay
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(config.retry_exceptions) as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        logger.warning(
                            f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{config.max_retries + 1}): {e}"
                        )
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        time.sleep(delay)
                        
                        if config.exponential_backoff:
                            delay = min(delay * 2, config.max_delay)
                    else:
                        logger.error(
                            f"函数 {func.__name__} 在 {config.max_retries + 1} 次尝试后仍然失败"
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


def validate_checkpoint_file(checkpoint_path: str) -> bool:
    """验证检查点文件的完整性。
    
    检查文件是否存在、是否可读、是否为有效的PyTorch检查点格式。
    
    Args:
        checkpoint_path: 检查点文件路径
        
    Returns:
        如果文件有效返回True，否则返回False
        
    Raises:
        CheckpointNotFoundError: 如果文件不存在
        CheckpointCorruptedError: 如果文件损坏
    """
    path = Path(checkpoint_path)
    
    # 检查文件是否存在
    if not path.exists():
        raise CheckpointNotFoundError(checkpoint_path)
    
    # 检查文件是否可读
    if not os.access(path, os.R_OK):
        raise CheckpointCorruptedError(checkpoint_path, "文件不可读")
    
    # 检查文件大小
    if path.stat().st_size == 0:
        raise CheckpointCorruptedError(checkpoint_path, "文件为空")
    
    # 尝试读取文件头部验证格式
    try:
        import torch
        # 尝试加载检查点（只加载元数据）
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 验证必需的键
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                raise CheckpointCorruptedError(
                    checkpoint_path, 
                    f"缺少必需的键: {key}"
                )
        
        return True
    except ImportError:
        # 如果没有安装PyTorch，跳过深度验证
        logger.warning("PyTorch未安装，跳过检查点深度验证")
        return True
    except Exception as e:
        if isinstance(e, (CheckpointNotFoundError, CheckpointCorruptedError)):
            raise
        raise CheckpointCorruptedError(checkpoint_path, str(e))


def check_disk_space(path: str, required_bytes: int) -> bool:
    """检查磁盘空间是否足够。
    
    Args:
        path: 要检查的路径
        required_bytes: 需要的字节数
        
    Returns:
        如果空间足够返回True
        
    Raises:
        DiskSpaceError: 如果空间不足
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        
        if free < required_bytes:
            raise DiskSpaceError(
                path=path,
                message=f"磁盘空间不足: 需要 {required_bytes / 1024 / 1024:.2f} MB，"
                        f"可用 {free / 1024 / 1024:.2f} MB",
                required_space=required_bytes,
                available_space=free
            )
        
        return True
    except OSError as e:
        raise DiskSpaceError(path=path, message=f"无法检查磁盘空间: {e}")


def check_gpu_availability() -> Tuple[bool, Optional[str]]:
    """检查GPU是否可用。
    
    Returns:
        元组 (是否可用, 设备名称或错误信息)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, device_name
        else:
            return False, "CUDA不可用"
    except ImportError:
        return False, "PyTorch未安装"
    except Exception as e:
        return False, str(e)


def get_device(prefer_gpu: bool = True) -> str:
    """获取计算设备，支持降级运行。
    
    如果请求GPU但不可用，自动降级到CPU。
    
    Args:
        prefer_gpu: 是否优先使用GPU
        
    Returns:
        设备字符串（'cuda' 或 'cpu'）
    """
    if prefer_gpu:
        available, info = check_gpu_availability()
        if available:
            logger.info(f"使用GPU: {info}")
            return 'cuda'
        else:
            logger.warning(f"GPU不可用 ({info})，降级到CPU")
            return 'cpu'
    else:
        return 'cpu'


class RecoveryManager:
    """错误恢复管理器。
    
    提供统一的错误恢复接口，包括：
    - 检查点恢复
    - 状态保存
    - 错误日志记录
    
    Attributes:
        checkpoint_dir: 检查点目录
        auto_save: 是否自动保存状态
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        auto_save: bool = True
    ):
        """初始化恢复管理器。
        
        Args:
            checkpoint_dir: 检查点目录
            auto_save: 是否自动保存状态
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_save = auto_save
        self._last_checkpoint: Optional[str] = None
        
        # 确保检查点目录存在
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的检查点文件。
        
        Returns:
            最新检查点的路径，如果没有则返回None
        """
        checkpoints = list(self.checkpoint_dir.glob('*.pt'))
        
        if not checkpoints:
            return None
        
        # 按修改时间排序，返回最新的
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def can_recover(self) -> bool:
        """检查是否可以从检查点恢复。
        
        Returns:
            如果存在有效的检查点返回True
        """
        latest = self.find_latest_checkpoint()
        if latest is None:
            return False
        
        try:
            validate_checkpoint_file(latest)
            return True
        except (CheckpointNotFoundError, CheckpointCorruptedError):
            return False
    
    def get_recovery_checkpoint(self) -> Optional[str]:
        """获取用于恢复的检查点路径。
        
        Returns:
            有效检查点的路径，如果没有则返回None
        """
        latest = self.find_latest_checkpoint()
        if latest is None:
            return None
        
        try:
            validate_checkpoint_file(latest)
            return latest
        except (CheckpointNotFoundError, CheckpointCorruptedError) as e:
            logger.warning(f"最新检查点无效: {e}")
            return None
    
    def record_error(
        self,
        error: Exception,
        context: Optional[dict] = None
    ) -> None:
        """记录错误信息。
        
        Args:
            error: 异常对象
            context: 额外的上下文信息
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        if isinstance(error, PokerAIError) and error.details:
            error_info['details'] = error.details
        
        logger.error(f"错误记录: {error_info}")
    
    def handle_training_interruption(
        self,
        save_callback: Optional[Callable[[], str]] = None,
        episode_number: Optional[int] = None
    ) -> TrainingInterruptedError:
        """处理训练中断。
        
        尝试保存当前状态，然后返回中断错误。
        
        Args:
            save_callback: 保存状态的回调函数，返回检查点路径
            episode_number: 当前回合数
            
        Returns:
            TrainingInterruptedError异常
        """
        checkpoint_saved = False
        
        if save_callback and self.auto_save:
            try:
                checkpoint_path = save_callback()
                self._last_checkpoint = checkpoint_path
                checkpoint_saved = True
                logger.info(f"训练中断，已保存检查点: {checkpoint_path}")
            except Exception as e:
                logger.error(f"保存检查点失败: {e}")
        
        return TrainingInterruptedError(
            message="训练被中断",
            episode_number=episode_number,
            checkpoint_saved=checkpoint_saved
        )


def graceful_shutdown(
    cleanup_callbacks: Optional[List[Callable[[], None]]] = None
) -> Callable:
    """优雅关闭装饰器。
    
    捕获KeyboardInterrupt和SystemExit，执行清理操作后退出。
    
    Args:
        cleanup_callbacks: 清理回调函数列表
        
    Returns:
        装饰器函数
        
    Example:
        @graceful_shutdown([save_state, close_connections])
        def main():
            # 主程序逻辑
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit) as e:
                logger.info("收到终止信号，正在执行清理操作...")
                
                if cleanup_callbacks:
                    for callback in cleanup_callbacks:
                        try:
                            callback()
                        except Exception as cleanup_error:
                            logger.error(f"清理操作失败: {cleanup_error}")
                
                logger.info("清理完成，程序退出")
                raise
        
        return wrapper
    
    return decorator


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    log_error: bool = True,
    **kwargs
) -> Optional[T]:
    """安全执行函数，捕获所有异常。
    
    Args:
        func: 要执行的函数
        *args: 位置参数
        default: 发生异常时的默认返回值
        log_error: 是否记录错误日志
        **kwargs: 关键字参数
        
    Returns:
        函数返回值或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
        return default
