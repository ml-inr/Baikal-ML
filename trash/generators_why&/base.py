import os

import numpy as np
import polars as pl

from data.process_manager.processor import Processor
from data.process_manager.config import ProcessorConfig
from data.settings.datasets_config import Paths2Root

class BaseGenerator:
    """
    По заданным путям до root файлов создаёт генератор датасета для обучения (data, labels).
    Это базовый класс: под каждую задачу создан свой генератор, наследуемый от данного.
    
    Args: 
        paths2root: list[str], 
        processor_cfg: ProcessorConfig
    """
    
    def __init__(self, paths2root: list[str], processor_cfg: ProcessorConfig):
        self.proc = Processor(paths2root, processor_cfg)
        self.df = self.proc.process()
        
    def gen_batch(self):
        ...