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
    def __post_init__(self):
        super().__post_init__()
        self.is_computational = True
        self.is_prime = True
  
@dataclass
class OperatorNonComputationalConfig(AbstractOperatorConfig):
    """
    """
    def __post_init__(self):
        super().__post_init__()
        self.is_computational = False
        self.is_prime = True

@dataclass
class OperatorCustomConfig(AbstractOperatorConfig):
    """
    """
    def __post_init__(self):
        super().__post_init__()
        self.is_computational = False
        self.is_prime = False