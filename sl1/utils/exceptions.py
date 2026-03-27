"""自定义异常类模块 - 定义系统中使用的所有自定义异常。

根据设计文档，系统需要处理以下类型的错误：
1. 配置错误 - 无效的参数值、缺失必需参数、不兼容的参数组合
2. 运行时错误 - 内存不足、磁盘空间不足、GPU不可用
3. 数据错误 - 检查点文件损坏、配置文件格式错误、日志文件写入失败
4. 游戏逻辑错误 - 非法行动尝试、状态不一致
5. 并行训练错误 - 工作进程崩溃、进程间通信失败
"""

from typing import Optional, List, Any


class PokerAIError(Exception):
    """德州扑克AI系统的基础异常类。
    
    所有自定义异常都继承自此类，便于统一捕获和处理。
    
    Attributes:
        message: 错误信息
        details: 额外的错误详情
    """
    
    def __init__(self, message: str, details: Optional[Any] = None):
        """初始化异常。
        
        Args:
            message: 错误信息
            details: 额外的错误详情（可选）
        """
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """返回异常的字符串表示。"""
        if self.details:
            return f"{self.message} (详情: {self.details})"
        return self.message


# ============== 配置错误 ==============

class ConfigurationError(PokerAIError):
    """配置错误的基类。
    
    当配置参数无效、缺失或不兼容时抛出。
    """
    pass


class InvalidParameterError(ConfigurationError):
    """无效参数错误。
    
    当配置参数的值无效时抛出（如负数学习率）。
    
    Attributes:
        parameter_name: 参数名称
        parameter_value: 参数值
        reason: 无效原因
    """
    
    def __init__(self, parameter_name: str, parameter_value: Any, reason: str):
        """初始化无效参数错误。
        
        Args:
            parameter_name: 参数名称
            parameter_value: 参数值
            reason: 无效原因
        """
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.reason = reason
        message = f"参数 '{parameter_name}' 无效: {reason} (当前值: {parameter_value})"
        super().__init__(message, details={
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "reason": reason
        })


class MissingParameterError(ConfigurationError):
    """缺失参数错误。
    
    当必需的配置参数缺失时抛出。
    
    Attributes:
        parameter_name: 缺失的参数名称
    """
    
    def __init__(self, parameter_name: str):
        """初始化缺失参数错误。
        
        Args:
            parameter_name: 缺失的参数名称
        """
        self.parameter_name = parameter_name
        message = f"缺失必需参数: '{parameter_name}'"
        super().__init__(message, details={"parameter_name": parameter_name})


class IncompatibleParametersError(ConfigurationError):
    """参数不兼容错误。
    
    当多个配置参数之间存在冲突时抛出。
    
    Attributes:
        parameters: 冲突的参数列表
        reason: 不兼容原因
    """
    
    def __init__(self, parameters: List[str], reason: str):
        """初始化参数不兼容错误。
        
        Args:
            parameters: 冲突的参数列表
            reason: 不兼容原因
        """
        self.parameters = parameters
        self.reason = reason
        params_str = ", ".join(f"'{p}'" for p in parameters)
        message = f"参数不兼容 ({params_str}): {reason}"
        super().__init__(message, details={
            "parameters": parameters,
            "reason": reason
        })


class ConfigValidationError(ConfigurationError):
    """配置验证错误。
    
    当配置验证失败时抛出，包含所有验证错误信息。
    
    Attributes:
        errors: 验证错误列表
    """
    
    def __init__(self, errors: List[str]):
        """初始化配置验证错误。
        
        Args:
            errors: 验证错误列表
        """
        self.errors = errors
        message = f"配置验证失败: {'; '.join(errors)}"
        super().__init__(message, details={"errors": errors})


# ============== 运行时错误 ==============

class RuntimeError(PokerAIError):
    """运行时错误的基类。
    
    当系统运行过程中发生错误时抛出。
    """
    pass


class ResourceError(RuntimeError):
    """资源错误的基类。
    
    当系统资源不足或不可用时抛出。
    """
    pass


class MemoryError(ResourceError):
    """内存不足错误。
    
    当系统内存不足时抛出。
    
    Attributes:
        required_memory: 需要的内存量（字节）
        available_memory: 可用的内存量（字节）
    """
    
    def __init__(
        self, 
        message: str = "内存不足",
        required_memory: Optional[int] = None,
        available_memory: Optional[int] = None
    ):
        """初始化内存不足错误。
        
        Args:
            message: 错误信息
            required_memory: 需要的内存量（字节）
            available_memory: 可用的内存量（字节）
        """
        self.required_memory = required_memory
        self.available_memory = available_memory
        details = {}
        if required_memory is not None:
            details["required_memory"] = required_memory
        if available_memory is not None:
            details["available_memory"] = available_memory
        super().__init__(message, details=details if details else None)


class DiskSpaceError(ResourceError):
    """磁盘空间不足错误。
    
    当磁盘空间不足时抛出。
    
    Attributes:
        path: 相关路径
        required_space: 需要的空间（字节）
        available_space: 可用的空间（字节）
    """
    
    def __init__(
        self,
        path: str,
        message: str = "磁盘空间不足",
        required_space: Optional[int] = None,
        available_space: Optional[int] = None
    ):
        """初始化磁盘空间不足错误。
        
        Args:
            path: 相关路径
            message: 错误信息
            required_space: 需要的空间（字节）
            available_space: 可用的空间（字节）
        """
        self.path = path
        self.required_space = required_space
        self.available_space = available_space
        details = {"path": path}
        if required_space is not None:
            details["required_space"] = required_space
        if available_space is not None:
            details["available_space"] = available_space
        super().__init__(message, details=details)


class GPUError(ResourceError):
    """GPU错误。
    
    当GPU不可用或发生GPU相关错误时抛出。
    
    Attributes:
        gpu_id: GPU设备ID
    """
    
    def __init__(self, message: str = "GPU不可用", gpu_id: Optional[int] = None):
        """初始化GPU错误。
        
        Args:
            message: 错误信息
            gpu_id: GPU设备ID
        """
        self.gpu_id = gpu_id
        details = {"gpu_id": gpu_id} if gpu_id is not None else None
        super().__init__(message, details=details)


class TrainingInterruptedError(RuntimeError):
    """训练中断错误。
    
    当训练被中断时抛出（如用户手动停止）。
    
    Attributes:
        episode_number: 中断时的回合数
        checkpoint_saved: 是否已保存检查点
    """
    
    def __init__(
        self,
        message: str = "训练被中断",
        episode_number: Optional[int] = None,
        checkpoint_saved: bool = False
    ):
        """初始化训练中断错误。
        
        Args:
            message: 错误信息
            episode_number: 中断时的回合数
            checkpoint_saved: 是否已保存检查点
        """
        self.episode_number = episode_number
        self.checkpoint_saved = checkpoint_saved
        details = {
            "episode_number": episode_number,
            "checkpoint_saved": checkpoint_saved
        }
        super().__init__(message, details=details)


# ============== 数据错误 ==============

class DataError(PokerAIError):
    """数据错误的基类。
    
    当数据处理过程中发生错误时抛出。
    """
    pass


class CheckpointError(DataError):
    """检查点错误的基类。
    
    当检查点操作失败时抛出。
    """
    pass


class CheckpointCorruptedError(CheckpointError):
    """检查点文件损坏错误。
    
    当检查点文件损坏或无法读取时抛出。
    
    Attributes:
        checkpoint_path: 检查点文件路径
    """
    
    def __init__(self, checkpoint_path: str, message: Optional[str] = None):
        """初始化检查点文件损坏错误。
        
        Args:
            checkpoint_path: 检查点文件路径
            message: 错误信息
        """
        self.checkpoint_path = checkpoint_path
        if message is None:
            message = f"检查点文件损坏: {checkpoint_path}"
        super().__init__(message, details={"checkpoint_path": checkpoint_path})


class CheckpointNotFoundError(CheckpointError):
    """检查点文件不存在错误。
    
    当请求的检查点文件不存在时抛出。
    
    Attributes:
        checkpoint_path: 检查点文件路径
    """
    
    def __init__(self, checkpoint_path: str):
        """初始化检查点文件不存在错误。
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        self.checkpoint_path = checkpoint_path
        message = f"检查点文件不存在: {checkpoint_path}"
        super().__init__(message, details={"checkpoint_path": checkpoint_path})


class ConfigFileError(DataError):
    """配置文件错误。
    
    当配置文件格式错误或无法解析时抛出。
    
    Attributes:
        file_path: 配置文件路径
    """
    
    def __init__(self, file_path: str, message: Optional[str] = None):
        """初始化配置文件错误。
        
        Args:
            file_path: 配置文件路径
            message: 错误信息
        """
        self.file_path = file_path
        if message is None:
            message = f"配置文件格式错误: {file_path}"
        super().__init__(message, details={"file_path": file_path})


class LogFileError(DataError):
    """日志文件错误。
    
    当日志文件写入或读取失败时抛出。
    
    Attributes:
        file_path: 日志文件路径
        operation: 操作类型（'read' 或 'write'）
    """
    
    def __init__(self, file_path: str, operation: str = "write", message: Optional[str] = None):
        """初始化日志文件错误。
        
        Args:
            file_path: 日志文件路径
            operation: 操作类型
            message: 错误信息
        """
        self.file_path = file_path
        self.operation = operation
        if message is None:
            message = f"日志文件{operation}失败: {file_path}"
        super().__init__(message, details={
            "file_path": file_path,
            "operation": operation
        })


class DataSerializationError(DataError):
    """数据序列化错误。
    
    当数据序列化或反序列化失败时抛出。
    
    Attributes:
        data_type: 数据类型
        operation: 操作类型（'serialize' 或 'deserialize'）
    """
    
    def __init__(self, data_type: str, operation: str = "serialize", message: Optional[str] = None):
        """初始化数据序列化错误。
        
        Args:
            data_type: 数据类型
            operation: 操作类型
            message: 错误信息
        """
        self.data_type = data_type
        self.operation = operation
        if message is None:
            op_str = "序列化" if operation == "serialize" else "反序列化"
            message = f"{data_type} {op_str}失败"
        super().__init__(message, details={
            "data_type": data_type,
            "operation": operation
        })


# ============== 游戏逻辑错误 ==============

class GameError(PokerAIError):
    """游戏逻辑错误的基类。
    
    当游戏逻辑出现问题时抛出。
    """
    pass


class IllegalActionError(GameError):
    """非法行动错误。
    
    当玩家尝试执行非法行动时抛出。
    
    Attributes:
        action: 尝试的行动
        reason: 非法原因
    """
    
    def __init__(self, action: Any, reason: str):
        """初始化非法行动错误。
        
        Args:
            action: 尝试的行动
            reason: 非法原因
        """
        self.action = action
        self.reason = reason
        message = f"非法行动: {reason}"
        super().__init__(message, details={
            "action": str(action),
            "reason": reason
        })


class InvalidGameStateError(GameError):
    """无效游戏状态错误。
    
    当游戏状态不一致或无效时抛出。
    
    Attributes:
        state_description: 状态描述
        reason: 无效原因
    """
    
    def __init__(self, state_description: str, reason: str):
        """初始化无效游戏状态错误。
        
        Args:
            state_description: 状态描述
            reason: 无效原因
        """
        self.state_description = state_description
        self.reason = reason
        message = f"无效游戏状态: {reason}"
        super().__init__(message, details={
            "state_description": state_description,
            "reason": reason
        })


class GameNotStartedError(GameError):
    """游戏未开始错误。
    
    当在游戏未开始时尝试执行游戏操作时抛出。
    """
    
    def __init__(self, message: str = "游戏尚未开始"):
        """初始化游戏未开始错误。
        
        Args:
            message: 错误信息
        """
        super().__init__(message)


class GameAlreadyEndedError(GameError):
    """游戏已结束错误。
    
    当在游戏已结束后尝试执行游戏操作时抛出。
    """
    
    def __init__(self, message: str = "游戏已结束"):
        """初始化游戏已结束错误。
        
        Args:
            message: 错误信息
        """
        super().__init__(message)


# ============== 并行训练错误 ==============

class ParallelTrainingError(PokerAIError):
    """并行训练错误的基类。
    
    当并行训练过程中发生错误时抛出。
    """
    pass


class WorkerProcessError(ParallelTrainingError):
    """工作进程错误。
    
    当工作进程崩溃或发生错误时抛出。
    
    Attributes:
        worker_id: 工作进程ID
        original_error: 原始错误
    """
    
    def __init__(self, worker_id: int, original_error: Optional[Exception] = None):
        """初始化工作进程错误。
        
        Args:
            worker_id: 工作进程ID
            original_error: 原始错误
        """
        self.worker_id = worker_id
        self.original_error = original_error
        message = f"工作进程 {worker_id} 发生错误"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message, details={
            "worker_id": worker_id,
            "original_error": str(original_error) if original_error else None
        })


class InterProcessCommunicationError(ParallelTrainingError):
    """进程间通信错误。
    
    当进程间通信失败时抛出。
    
    Attributes:
        source_process: 源进程ID
        target_process: 目标进程ID
    """
    
    def __init__(
        self,
        message: str = "进程间通信失败",
        source_process: Optional[int] = None,
        target_process: Optional[int] = None
    ):
        """初始化进程间通信错误。
        
        Args:
            message: 错误信息
            source_process: 源进程ID
            target_process: 目标进程ID
        """
        self.source_process = source_process
        self.target_process = target_process
        details = {}
        if source_process is not None:
            details["source_process"] = source_process
        if target_process is not None:
            details["target_process"] = target_process
        super().__init__(message, details=details if details else None)


class ParameterSyncError(ParallelTrainingError):
    """参数同步错误。
    
    当参数同步失败时抛出。
    """
    
    def __init__(self, message: str = "参数同步失败"):
        """初始化参数同步错误。
        
        Args:
            message: 错误信息
        """
        super().__init__(message)


# ============== 模型错误 ==============

class ModelError(PokerAIError):
    """模型错误的基类。
    
    当模型操作失败时抛出。
    """
    pass


class ModelLoadError(ModelError):
    """模型加载错误。
    
    当模型加载失败时抛出。
    
    Attributes:
        model_path: 模型路径
    """
    
    def __init__(self, model_path: str, message: Optional[str] = None):
        """初始化模型加载错误。
        
        Args:
            model_path: 模型路径
            message: 错误信息
        """
        self.model_path = model_path
        if message is None:
            message = f"模型加载失败: {model_path}"
        super().__init__(message, details={"model_path": model_path})


class ModelSaveError(ModelError):
    """模型保存错误。
    
    当模型保存失败时抛出。
    
    Attributes:
        model_path: 模型路径
    """
    
    def __init__(self, model_path: str, message: Optional[str] = None):
        """初始化模型保存错误。
        
        Args:
            model_path: 模型路径
            message: 错误信息
        """
        self.model_path = model_path
        if message is None:
            message = f"模型保存失败: {model_path}"
        super().__init__(message, details={"model_path": model_path})


# ============== 评估错误 ==============

class EvaluationError(PokerAIError):
    """评估错误的基类。
    
    当模型评估过程中发生错误时抛出。
    """
    pass


class InvalidOpponentError(EvaluationError):
    """无效对手错误。
    
    当指定的对手策略无效时抛出。
    
    Attributes:
        opponent_name: 对手名称
    """
    
    def __init__(self, opponent_name: str):
        """初始化无效对手错误。
        
        Args:
            opponent_name: 对手名称
        """
        self.opponent_name = opponent_name
        message = f"无效的对手策略: {opponent_name}"
        super().__init__(message, details={"opponent_name": opponent_name})
