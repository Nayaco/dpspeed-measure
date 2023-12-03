# Copyright (c) 2023, ISCS, Wenjie Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler

from dataclasses import dataclass
from typing import Callable, Tuple
from abc import abstractmethod, ABC

@dataclass
class AbstractOperatorConfig:
    """
    
    """
    op_uid: int = 0
    op_name: str = None

    is_computational = False
    is_prime = False

    # computational_density: float = 0.0
    # computational_latency: float = 0.0

class AbstractOperator(ABC):
    """
    Define the operators as Graph G<V,E> where V are operators, E define 
    dependency between contiguous operator excutions. Sibling operations
    can be well paralleled.
    Flatten the Graph G<V,E> to a list of operators T<op1, op2, ...>
    """
    def __init__(self, config: AbstractOperatorConfig):
        self._config = config
        # Prime Operator Only
        self._next: list[AbstractOperator] = [] 
        self._prev: list[AbstractOperator] = []
        self._prev_done: int = int(0)

    def add_next(self, next_op):
        """
        only when the operator is prime, it can add next_op like that
        should be override when operator is not prime
        """
        if isinstance(next_op, AbstractOperator):
            self._next.append(next_op)
            next_op._prev.append(self)
        else:
            raise TypeError("AbstractOperator.add_next() accept next_op should be AbstractOperator")
    
    @abstractmethod
    def estimate(self, *tensor_in: torch.Tensor) -> int:
        """
        *tensor_in: tensor shapes
        return operator excution time
        """    
        pass
    
    def reset(self) -> None:
        self._prev_done = int(0)
        
    def default_apply_cb(self):
        """
        default callback function for apply
        """
        for _op in self._next:
            _op._prev_done += 1

    @abstractmethod
    def apply(self) -> bool:
        pass
    
    def __repr__(self) -> str:
        return "<operator-%d, %s>" % (self._config.op_uid, '{0: <5}'.format(self._config.op_name or 'None'))
    
    def __len__(self) -> int:
        return 0 if self._next is None else len(self._next)

@dataclass
class OperatorComputationalConfig(AbstractOperatorConfig):
    """
    """
    def __post_init__(self):
        self.is_computational = True
        self.is_prime = True
  
@dataclass
class OperatorNonComputationalConfig(AbstractOperatorConfig):
    """
    """
    def __post_init__(self):
        self.is_computational = False
        self.is_prime = True

@dataclass
class OperatorCustomConfig(AbstractOperatorConfig):
    """
    """
    def __post_init__(self):
        self.is_computational = False
        self.is_prime = False

class OpStaticComputational(AbstractOperator):
    """
    static computational : 
    """
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)
    
    def estimate(self, *tensor_in: torch.Tensor) -> int:
        return 0
    
class OpStaticNonComputational(AbstractOperator):
    """
    static non computational : 
    """
    def __init__(self, config: OperatorNonComputationalConfig):
        super().__init__(config)
    
    def estimate(self, *tensor_in: torch.Tensor) -> int:
        return 0
    
class OpStaticDerivative(AbstractOperator):
    """
    static derivative operator : 
    """
    def __init__(self, config: OperatorCustomConfig):
        super().__init__(config)
        self._subop : list[AbstractOperator] = None
    
    def estimate(self, *tensor_in: torch.Tensor) -> int:
        return 0

    def apply(self) -> bool:
        """
        apply operator
        """
        assert self._subop is not None and len(self._subop) > 0, "OpStaticDerivative._subop is None"
        return self._subop[0].apply()
    
    def reset(self) -> None:
        super().reset()
        if self._subop is not None:
            for _op in self._subop:
                _op.reset()
        
    def subop(self):
        """
        only when the operator is not prime, it can have subop, return the head of subops
        """
        return self._subop