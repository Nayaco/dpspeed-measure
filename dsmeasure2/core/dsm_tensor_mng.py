# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any
from functools import cache

import torch
import torch.nn.functional as F

from dsmeasure2.core.dsm_tensor import AbstractTensor

def gen_tensor_uid() -> int:
    op_uid: int = int(64) # from 0x40
    while True:
        yield op_uid
        op_uid += 1
IDGenerator = gen_tensor_uid()
@cache
class TensorManager:
    def __init__(self) -> None:
        self.tensors: dict[int, AbstractTensor] = {}
        self.tensor_count: int = 0

    def register(self, _T: AbstractTensor) -> AbstractTensor:
        """
        register tensor
        """
        _T.tensor_uid = next(IDGenerator)
        self.tensors[_T.tensor_uid] = _T
        return self.tensors[_T.tensor_uid]
        
    def find(self, _uid: int) -> AbstractTensor:
        """
        find tensor
        return: (tensor,)
        """
        return self.tensors[_uid]
    
    def __iter__(self):
        return iter(self.tensors.values())