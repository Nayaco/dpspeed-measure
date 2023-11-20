# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_operator import AbstractOperator, AbstractOperatorConfig
from dsmeasure.common_operators.op_config import OperatorComputationalConfig

class OpSoftmax(AbstractOperator):
    """
    add: (b * h1 * h2) -> (b * h1 * h2)
        *tensor_baseline: [Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        # b * h1 * h2
        self.scale_baseline = torch.prod(tensor_baseline[0], 0)
   
    def estimate(self, *tensor_in: torch.Tensor) -> tuple(int, tuple):
        """
        estimate add operator 
            *tensor_in: [Tensor(3)]
        return (int, Tensor(3))
            run_time
            tensor_shape
        """
        scale = torch.prod(tensor_in[0], 0)
        return int(torch.Tensor(self.time_baseline, dtype=float) * (scale / self.scale_baseline)), \
                tensor_in[0]

class OpReLUGeLU(AbstractOperator):
    """
    add: (b * h1 * h2) -> (b * h1 * h2)
        *tensor_baseline: [Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        # b * h1 * h2
        self.scale_baseline = torch.prod(tensor_baseline[0], 0)
   
    def estimate(self, *tensor_in: torch.Tensor) -> tuple(int, tuple):
        """
        estimate add operator 
            *tensor_in: [Tensor(3)]
        return (int, Tensor(3))
            run_time
            tensor_shape
        """
        scale = torch.prod(tensor_in[0], 0)
        return int(torch.Tensor(self.time_baseline, dtype=float) * (scale / self.scale_baseline)), \
                tensor_in[0]
