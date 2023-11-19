# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_operator import AbstractOperator, AbstractOperatorConfig
from .op_config import OperatorComputationalConfig

class OpBaddmm(AbstractOperator):
    """
    baddmm: (b * h1 * h2) + (b * h1 * w) * (b * w * h2) -> (b * h1 * h2)
        *tensor_baseline: [Tensor(3), Tensor(3), Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        # b * h1 * w * h2
        self.scale_baseline = \
            tensor_baseline[1][0] * tensor_baseline[1][1] * tensor_baseline[1][2] * tensor_baseline[2][2]
    """
    Apply addmm operator
        *tensor_in: [Tensor(3), Tensor(3), Tensor(3)]
    return (int, Tensor(3))
        run_time
        tensor_shape
    """
    def apply(self, *tensor_in: torch.Tensor) -> tuple(int, torch.Tensor):
        scale = tensor_in[1][0] * tensor_in[1][1] * tensor_in[1][2] * tensor_in[2][2]
        return int(torch.Tensor(self.time_baseline, dtype=float) * (scale / self.scale_baseline)), \
                tensor_in[0]

class OpAdd(AbstractOperator):
    """
    add: (b * h1 * h2) + (b * h1 * h2) -> (b * h1 * h2)
        *tensor_baseline: [Tensor(3), Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        # b * h1 * h2
        self.scale_baseline = torch.dot(tensor_baseline[0])
    """
    Apply add operator 
        *tensor_in: [Tensor(3), Tensor(3)]
    return (int, Tensor(3))
        run_time
        tensor_shape
    """
    def apply(self, *tensor_in: torch.Tensor) -> tuple(int, tuple):
        scale = torch.dot(tensor_in[0])
        return int(torch.Tensor(self.time_baseline, dtype=float) * (scale / self.scale_baseline)), \
                tensor_in[0]