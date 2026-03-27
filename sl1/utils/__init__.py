"""工具函数和辅助模块。

该包提供以下功能：
- 配置管理（ConfigManager）
- 数据日志（DataLogger）
- 检查点管理（CheckpointManager）
- 自定义异常（exceptions）
- 日志记录（logger）
- 错误恢复（error_recovery）
"""

from utils.config_manager import ConfigManager, DEFAULT_CONFIG
from utils.data_logger import DataLogger
from utils.checkpoint_manager import CheckpointManager

# 导出异常类
from utils.exceptions import (
    # 基础异常
    PokerAIError,
    # 配置错误
    ConfigurationError,
    InvalidParameterError,
    MissingParameterError,
    IncompatibleParametersError,
    ConfigValidationError,
    # 运行时错误
    ResourceError,
    MemoryError,
    DiskSpaceError,
    GPUError,
    TrainingInterruptedError,
    # 数据错误
    DataError,
    CheckpointError,
    CheckpointCorruptedError,
    CheckpointNotFoundError,
    ConfigFileError,
    LogFileError,
    DataSerializationError,
    # 游戏逻辑错误
    GameError,
    IllegalActionError,
    InvalidGameStateError,
    GameNotStartedError,
    GameAlreadyEndedError,
    # 并行训练错误
    ParallelTrainingError,
    WorkerProcessError,
    InterProcessCommunicationError,
    ParameterSyncError,
    # 模型错误
    ModelError,
    ModelLoadError,
    ModelSaveError,
    # 评估错误
    EvaluationError,
    InvalidOpponentError,
)

# 导出日志功能
from utils.logger import (
    LoggerConfig,
    setup_logger,
    get_logger,
    set_log_level,
    clear_loggers,
    LoggerMixin,
    configure_logging,
    get_global_config,
    get_training_logger,
    get_environment_logger,
    get_model_logger,
    get_monitoring_logger,
    get_analysis_logger,
    get_utils_logger,
)

# 导出错误恢复功能
from utils.error_recovery import (
    RetryConfig,
    retry,
    validate_checkpoint_file,
    check_disk_space,
    check_gpu_availability,
    get_device,
    RecoveryManager,
    graceful_shutdown,
    safe_execute,
)

__all__ = [
    # 配置管理
    'ConfigManager',
    'DEFAULT_CONFIG',
    # 数据日志
    'DataLogger',
    # 检查点管理
    'CheckpointManager',
    # 基础异常
    'PokerAIError',
    # 配置错误
    'ConfigurationError',
    'InvalidParameterError',
    'MissingParameterError',
    'IncompatibleParametersError',
    'ConfigValidationError',
    # 运行时错误
    'ResourceError',
    'MemoryError',
    'DiskSpaceError',
    'GPUError',
    'TrainingInterruptedError',
    # 数据错误
    'DataError',
    'CheckpointError',
    'CheckpointCorruptedError',
    'CheckpointNotFoundError',
    'ConfigFileError',
    'LogFileError',
    'DataSerializationError',
    # 游戏逻辑错误
    'GameError',
    'IllegalActionError',
    'InvalidGameStateError',
    'GameNotStartedError',
    'GameAlreadyEndedError',
    # 并行训练错误
    'ParallelTrainingError',
    'WorkerProcessError',
    'InterProcessCommunicationError',
    'ParameterSyncError',
    # 模型错误
    'ModelError',
    'ModelLoadError',
    'ModelSaveError',
    # 评估错误
    'EvaluationError',
    'InvalidOpponentError',
    # 日志功能
    'LoggerConfig',
    'setup_logger',
    'get_logger',
    'set_log_level',
    'clear_loggers',
    'LoggerMixin',
    'configure_logging',
    'get_global_config',
    'get_training_logger',
    'get_environment_logger',
    'get_model_logger',
    'get_monitoring_logger',
    'get_analysis_logger',
    'get_utils_logger',
    # 错误恢复功能
    'RetryConfig',
    'retry',
    'validate_checkpoint_file',
    'check_disk_space',
    'check_gpu_availability',
    'get_device',
    'RecoveryManager',
    'graceful_shutdown',
    'safe_execute',
]
