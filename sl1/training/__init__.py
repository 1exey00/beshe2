"""训练模块导出（当前保留 CFR+SL 主流程所需组件）。"""

from training.cfr_trainer import CFRTrainer, InfoSet
from training.training_engine import TrainingEngine

__all__ = [
    "CFRTrainer",
    "InfoSet",
    "TrainingEngine",
]
