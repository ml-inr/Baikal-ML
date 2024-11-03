from typing import Optional, Dict, Any

from dataclasses import dataclass, field

from ..base_config import BaseConfig


@dataclass
class OptimizerConfig(BaseConfig):
    name: str = 'Adam'
    kwargs: dict = field(default_factory = lambda: {'lr': 0.001, 'weight_decay': 1e-5})
    
@dataclass
class SchedulerConfig(BaseConfig):
    name: str = 'StepLR'
    kwargs: Dict[str, Any]= field(default_factory = lambda: {'step_size': 10,'gamma': 0.1})

@dataclass
class LossConfig(BaseConfig):
    name: str = 'FocalLoss'
    kwargs: Dict[str, Any] = field(default_factory = lambda: {'alpha': 0.25, 'gamma': 2})

@dataclass
class TrainerConfig(BaseConfig):
    num_of_epochs: int = 3
    steps_per_epoch: Optional[int] = 500 # num of batches per epoch
    
    checkpoint_interval: int = 1 # save model every ... epochs
    checkpoint_path: str = "/home/albert/Baikal-ML/experiments/testing"
    log_interval: int = 100 # make logs of metrics every ... steps
    
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: Optional[SchedulerConfig] = None #SchedulerConfig()
    loss: LossConfig = LossConfig()