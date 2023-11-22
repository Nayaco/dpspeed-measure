# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_operator import AbstractOperator, AbstractOperatorConfig
from dsmeasure.common_operators.op_config import OperatorComputationalConfig, OperatorNonComputationalConfig

class OpLayerNorm(AbstractOperator):
    """
    add: (b * h1 * h2) -> (b * h1 * h2)
        *tensor_baseline: [Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        # b * h1 * h2
        self.scale_baseline = torch.prod(tensor_baseline[0], 0)
   
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
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

class OpAdd(AbstractOperator):
    """
    add: (b * h1 * h2) + (b * h1 * h2) -> (b * h1 * h2)
        *tensor_baseline: [Tensor(3), Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        # b * h1 * h2
        self.scale_baseline = torch.prod(tensor_baseline[0], 0)
   
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        estimate add operator 
            *tensor_in: [Tensor(3), Tensor(3)]
        return (int, Tensor(3))
            run_time
            tensor_shape
        """
        scale = torch.prod(tensor_in[0], 0)
        return int(torch.Tensor(self.time_baseline, dtype=float) * (scale / self.scale_baseline)), \
                tensor_in[0]

class OpMatmul(AbstractOperator):
    """
    matmul: (h1 * w) * (w * h2) -> (h1 * h2)
        *tensor_baseline: [Tensor(2), Tensor(2)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        # h1 * w * h2
        self.scale_baseline = \
            tensor_baseline[0][0] * tensor_baseline[0][1] * tensor_baseline[1][1]
    
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        estimate matmul operator
            *tensor_in: [Tensor(2), Tensor(2)]
        return (int, Tensor(2))
            run_time
            tensor_shape
        """
        scale = tensor_in[0][0] * tensor_in[0][1] * tensor_in[1][1]
        return int(torch.Tensor(self.time_baseline, dtype=float) * (scale / self.scale_baseline)), \
                torch.Tensor([tensor_in[0][0],tensor_in[1][1]]) 
    
class OpAddmm(AbstractOperator):
    """
    baddmm: (h1 * h2) + (h1 * w) * (w * h2) -> (h1 * h2)
        *tensor_baseline: [Tensor(3), Tensor(3), Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.mm = OpMatmul(config, time_baseline, tensor_baseline[1], tensor_baseline[2])
    
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        estimate addmm operator
            *tensor_in: [Tensor(3), Tensor(3), Tensor(3)]
        return (int, Tensor(3))
            run_time
            tensor_shape
        """
        return self.mm.estimate(tensor_in[1], tensor_in[2])
    
class OpBmm(AbstractOperator):
    """
    bmm: (b * h1 * w) * (b * w * h2) -> (b * h1 * h2)
        *tensor_baseline: [Tensor(3), Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.time_baseline = time_baseline
        self.scale_baseline = \
            tensor_baseline[0][0] * tensor_baseline[0][1] * tensor_baseline[1][1] * tensor_baseline[1][2]
    
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        estimate bmm operator
            *tensor_in: [Tensor(3), Tensor(3)]
        return (int, Tensor(3))
            run_time
            tensor_shape
        """
        scale = tensor_in[0][0] * tensor_in[0][1] * tensor_in[1][1] * tensor_in[1][2]
        return int(torch.Tensor(self.time_baseline, dtype=float) * (scale / self.scale_baseline)), \
                torch.Tensor([tensor_in[0][0], tensor_in[0][1], tensor_in[1][2]])
    
class OpBaddbmm(AbstractOperator):
    """
    baddmm: (b * h1 * h2) + (b * h1 * w) * (b * w * h2) -> (b * h1 * h2)
        *tensor_baseline: [Tensor(3), Tensor(3), Tensor(3)]
    """
    def __init__(self, config: OperatorComputationalConfig, time_baseline: int, *tensor_baseline: torch.Tensor):
        super().__init__(config)
        self.bmm = OpBmm(config, time_baseline, tensor_baseline[1], tensor_baseline[2])
    
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        estimate baddmm operator
            *tensor_in: [Tensor(3), Tensor(3), Tensor(3)]
        return (int, Tensor(3))
            run_time
            tensor_shape
        """
        return self.bmm.estimate(tensor_in[1], tensor_in[2])
    
class OpStaticComputational(AbstractOperator):
    """
    static computational : 
        *tensor_baseline: [Tensor(any)]
    """
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)
        # self.time_baseline = time_baseline
        # self.scale_baseline = None tensor_baseline[0]
    
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        estimate operator
            *tensor_in: [any]
        return (int, Tensor(any))
            run_time
            tensor_shape
        """
        return 0, None # self.time_baseline, self.scale_baseline
    
class OpStaticNonComputational(AbstractOperator):
    """
    static non computational : 
        *tensor_baseline: [Tensor(any)]
    """
    def __init__(self, config: OperatorNonComputationalConfig):
        super().__init__(config)
        # self.time_baseline = time_baseline
        # self.scale_baseline = tensor_baseline[0]
    
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        estimate operator
            *tensor_in: [any]
        return (int, Tensor(any))
            run_time
            tensor_shape
        """
        return 0, None # self.time_baseline, self.scale_baseline


