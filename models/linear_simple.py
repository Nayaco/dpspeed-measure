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

from dsmeasure.core.abstract_operator import AbstractOperatorConfig, AbstractOperator
from dsmeasure.common_operators.op_common import OpStaticComputational, OpStaticNonComputational
from dsmeasure.common_operators.op_config import OperatorComputationalConfig, OperatorNonComputationalConfig, OperatorCustomConfig


class Linear2Layer(AbstractOperator):
    def __init__(self, config: OperatorCustomConfig):
        super().__init__(config)
        self.matul_1 = OpStaticComputational(OperatorComputationalConfig(
            op_uid=1,
            op_name="matul",
        ), 100, torch.Tensor([1,2,3]))
        self.matul_2 = OpStaticComputational(OperatorComputationalConfig(
            op_uid=2,
            op_name="matul",
        ), 100, torch.Tensor([1,2,3]))

        # self.matul_1_backward = OpStaticComputational(OperatorComputationalConfig(
        #     op_uid=3,
        #     op_name="matul_backward",
        # ), 100, torch.Tensor([1,2,3]))
        # self.matul_2_backward = OpStaticComputational(OperatorComputationalConfig(
        #     op_uid=4,
        #     op_name="matul_backward",
        # ), 100, torch.Tensor([1,2,3]))
        self.matul_1.add_next(self.matul_2)
        self._subop = self.matul_1
    
    def estimate(self, *tensor_in: torch.Tensor):
        pass

    def __repr__(self) -> str:
        prefix = f'{self.matul_1.__repr__()}---{str(self.matul_2.__repr__())}'
        return '%s\n%s%s' % (super().__repr__(), ' '*(len(super().__repr__())) ,prefix)