# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any
from functools import cache

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_operator import AbstractOperatorConfig, AbstractOperator
from dsmeasure.common_operators.op_config import OperatorComputationalConfig, OperatorNonComputationalConfig, OperatorCustomConfig

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

    def register(self, new_op: AbstractOperator) -> tuple[int, AbstractOperator]:
        """
        register operator:
            new_op: op
        return: (op_uid, op)
        """
        head_operator_uid = None
        new_op_queue: list[AbstractOperator] = [new_op]
        while len(new_op_queue) > 0:
            new_operator_uid = next(IDGenerator)
            self.operators[new_operator_uid] = new_op_queue[0]
            self.operators[new_operator_uid]._config.op_uid = new_operator_uid
            self.op_count += 1
            
            if new_op_queue[0]._config.is_prime:
                new_op_queue.extend(new_op_queue[0]._next)
            else:
                new_op_queue.append(new_op_queue[0].subop())
            new_op_queue.pop(0)
            if head_operator_uid is None:
                head_operator_uid = new_operator_uid
        
        return head_operator_uid, self.operators[head_operator_uid]
        
    def find(self, operator_uid: int) -> AbstractOperator:
        """
        find device:
            device_uid: device uid
        return: (device,)
        """
        return self.operators[operator_uid]
    
    def __iter__(self):
        return iter(self.operators.values())