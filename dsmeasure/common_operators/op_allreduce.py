# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_operator import AbstractOperator, AbstractOperatorConfig
from dsmeasure.common_operators.op_config import OperatorNonComputationalConfig

class OpAllRecuce(AbstractOperator):
    """
    allreduce: (s) -> (s)
    """
    def __init__(self, config: OperatorNonComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        # tensor size
        self.scale_baseline = torch.prod(tensor_baseline[0], 0)
    
    def estimate(self, *tensor_in: torch.Tensor) -> tuple(int, torch.Tensor):
        """
        estimate allreduce operator
        """
        scale = torch.prod(tensor_in[0], 0)
        return int(torch.Tensor(self.time_baseline, dtype=float) * (scale / self.scale_baseline)), \
                tensor_in[0]