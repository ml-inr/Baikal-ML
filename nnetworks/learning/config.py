from typing import Optional

from dataclasses import dataclass

from ..configurations.base import BaseConfig


@dataclass
class OptimizerConfig:
    name: str = 'Adam'
    kwargs: dict = {'lr': 0.001, 'weight_decay': 1e-5}
    
@dataclass
class SchedulerConfig:
    name: str = 'StepLR'
    kwargs: dict = {'step_size': 10,'gamma': 0.1}

@dataclass
class LossConfig:
    name: str = 'FocalLoss'
    kwargs: dict = {'alpha': 0.25, 'gamma': 2}

@dataclass
class TrainerConfig:
    num_of_epochs: int = 10
    # num of steps to write logs after
    log_interval: int = 100 
    steps_per_epoch: Optional[int] = 10_000
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    loss: LossConfig = LossConfig()
    