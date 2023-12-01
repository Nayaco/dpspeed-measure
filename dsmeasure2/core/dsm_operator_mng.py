# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any
from functools import cache

import torch
import torch.nn.functional as F

from dsmeasure2.core.dsm_operator import AbstractOperatorConfig, \
                                         AbstractOperator, \
                                         OperatorComputationalConfig, \
                                         OperatorNonComputationalConfig , \
                                         OperatorCustomConfig , \
                                         OpStaticComputational , \
                                         OpStaticNonComputational, \
                                         OpStaticDerivative

def gen_operator_uid() -> int:
    op_uid: int = int(64) # from 0x40
    while True:
        yield op_uid
        op_uid += 1
IDGenerator = gen_operator_uid()
@cache
class OperatorManager:
    def __init__(self) -> None:
        self.operators: dict[int, AbstractOperator] = {}
        self.op_count: int = 0

    def register(self, _op: AbstractOperator) -> None:
        """
        register operator:
        """
        _op._config.op_uid = next(IDGenerator)
        self.operators[_op._config.op_uid] = _op
        self.op_count += 1
        if isinstance(_op, OpStaticDerivative):
            for _subop in _op._subop:
                self.register(_subop)
        
    def find(self, operator_uid: int) -> AbstractOperator:
        """
        find operator:
        """
        return self.operators[operator_uid]
    
    def __iter__(self):
        return iter(self.operators.values())