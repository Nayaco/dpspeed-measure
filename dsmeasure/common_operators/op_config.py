# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_operator import AbstractOperatorConfig

@dataclass
class OperatorComputationalConfig(AbstractOperatorConfig):
    """
    """
    operator_device_uid: list[int] = []
    
    def __post_init__(self):
        super().__post_init__()
        self.is_computational = True
  
@dataclass
class OperatorNonComputationalConfig(AbstractOperatorConfig):
    """
    """
    operator_device_uid: list[int] = []
    
    def __post_init__(self):
        super().__post_init__()
        self.is_computational = False