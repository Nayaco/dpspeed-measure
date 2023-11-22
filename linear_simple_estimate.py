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
from dsmeasure.core.operator_manager import OperatorManager
from dsmeasure.core.device_manager import DeviceManager
from dsmeasure.device.gpu import DeviceCUDAConfig
from models import linear_simple
from models import layer2linear

from dsmeasure.core.engine import CostEngine

# ls_ = linear_simple.Linear2Layer(OperatorCustomConfig(op_uid=0, op_name="linear_simple"))
# _oid, _op = OperatorManager().register(ls_)
# print(_oid)
# print(OperatorManager().find(_oid))

layer = layer2linear.Linear2Network(OperatorCustomConfig(op_uid=0, op_name="layer2linear"))
OperatorManager().register(layer)
print(layer)

DeviceManager().register(DeviceCUDAConfig(memory_max_capacity=1000, memory_limit_capacity=1000))

CostEngine().evaluation(10, [layer._config.op_uid])