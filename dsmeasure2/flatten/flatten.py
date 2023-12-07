# Copyright (c) 2023, ISCS, Wenjie Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import builtins
import datetime
import json
import sys
import time

import numpy as np
import torch

from dsmeasure2.core.dsm_device_mng import DeviceManager
from dsmeasure2.core.dsm_operator_mng import OperatorManager

from dsmeasure2.core.dsm_tensor import AbstractTensor
from dsmeasure2.core.dsm_device import AbstractDeviceConfig, AbstractDevice
from dsmeasure2.core.dsm_operator import AbstractOperatorConfig, \
                                         AbstractOperator, \
                                         OperatorComputationalConfig, \
                                         OperatorNonComputationalConfig , \
                                         OperatorCustomConfig , \
                                         OpStaticComputational , \
                                         OpStaticNonComputational, \
                                         OpStaticDerivative

from dsmeasure2.device.device_cuda import DeviceCUDA, DeviceCUDAConfig
from dsmeasure2.device.device_pcie import DevicePCIE4, DevicePCIEConfig

from dsmeasure2.core.dsm_tensor_mng import TensorManager
from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState
from dsmeasure2.graph.operator_graph import UnaryOperator, BinaryOperator, TernaryOperator, InitiateOperator

from dsmeasure2.flatten.flatten_operator import FlattenOperator, FlattenInitiate


def _flatten(operators: list[int], graph_head: list[int] = [0]):
    op_manager = OperatorManager()
    
    ready_queue: list[int] = [operators[_head] for _head in graph_head]
    op_seq: list[int] = []
    while len(ready_queue) > 0:
        _op_uid = ready_queue.pop(0)
        if not op_manager.operators[_op_uid]._config.is_prime:
            op_seq.extend(_flatten([__op._config.op_uid for __op in op_manager.operators[_op_uid].subop()], [0]))
        else:
            op_seq.append(_op_uid)
        
        for n_op in op_manager.operators[_op_uid]._next:
            n_op._prev_done += 1
            if n_op._prev_done == len(n_op._prev):
                ready_queue.append(n_op._config.op_uid)
    return op_seq

def flatten(operators: list[int], graph_head: list[int] = [0], assign_next = False):
    """
    Flatten the Graph G<V,E> to a list of operators T<op1, op2, ...> in graph operator sequence
    """
    op_manager = OperatorManager()
    for _op_uid in operators:
        op_manager.operators[_op_uid].reset()
    op_seq = _flatten(operators, graph_head)
    if assign_next:
        for i in range(len(op_seq)-1):
            op_manager.operators[op_seq[i]]._next = [op_manager.operators[op_seq[i+1]]]
            op_manager.operators[op_seq[i+1]]._prev = [op_manager.operators[op_seq[i]]]
        op_manager.operators[op_seq[i+1]]._next = []
    return op_seq

def convert_graph_to_flatten_seq(operators: list[int], graph_head: list[int] = [0]):
    """
    Convert the G<V, E> to a list of operators T<op1, op2, ...> in flatten operator sequence
    """
    _flatten_op_seq = flatten(operators, graph_head)
    _flatten_op_seq_ret = []
    o_mng = OperatorManager()
    t_mng = TensorManager()
    for _op_uid in _flatten_op_seq:
        _flatten_op: FlattenOperator|FlattenInitiate = \
                o_mng.register(
                    FlattenOperator(OperatorComputationalConfig(op_name=(o_mng.operators[_op_uid]._config.op_name+'.')[:-1])) ) \
            if not isinstance(o_mng.operators[_op_uid], InitiateOperator) else \
                o_mng.register(
                    FlattenInitiate(OperatorComputationalConfig(op_name=(o_mng.operators[_op_uid]._config.op_name+'.')[:-1])) )
        _flatten_op_seq_ret.append(_flatten_op._config.op_uid)
        if isinstance(o_mng.operators[_op_uid], UnaryOperator):
            _flatten_op.input = [t_mng.register(
                    ActivationTensor(tensor_size=o_mng.operators[_op_uid].input.tensor_size) ),]
            _flatten_op.output = [t_mng.register(
                    ActivationTensor(tensor_size=_output.tensor_size) ) 
                for _output in o_mng.operators[_op_uid].output ]
            _flatten_op.weight = None if o_mng.operators[_op_uid].weight is None else t_mng.register(
                    WeightTensor(tensor_size=o_mng.operators[_op_uid].weight.tensor_size) )
            _flatten_op.intermediate_memory = o_mng.operators[_op_uid].intermediate_memory
            _flatten_op.estimate_runtime = o_mng.operators[_op_uid].estimate_runtime
            _flatten_op.device_name = o_mng.operators[_op_uid].device_name.copy()
            _flatten_op.callback = None
        elif isinstance(o_mng.operators[_op_uid], BinaryOperator):
            _flatten_op.input = [t_mng.register(
                    ActivationTensor(tensor_size=o_mng.operators[_op_uid].input_a.tensor_size) ),
                                 t_mng.register(
                    ActivationTensor(tensor_size=o_mng.operators[_op_uid].input_b.tensor_size) ),]
            _flatten_op.output = [t_mng.register(
                    ActivationTensor(tensor_size=_output.tensor_size) ) 
                for _output in o_mng.operators[_op_uid].output ]
            _flatten_op.weight = None if o_mng.operators[_op_uid].weight is None else t_mng.register(
                    WeightTensor(tensor_size=o_mng.operators[_op_uid].weight.tensor_size) )
            _flatten_op.intermediate_memory = o_mng.operators[_op_uid].intermediate_memory
            _flatten_op.estimate_runtime = o_mng.operators[_op_uid].estimate_runtime
            _flatten_op.device_name = o_mng.operators[_op_uid].device_name.copy()
            _flatten_op.callback = None
        elif isinstance(o_mng.operators[_op_uid], TernaryOperator):
            _flatten_op.input = [t_mng.register(
                    ActivationTensor(tensor_size=o_mng.operators[_op_uid].input_a.tensor_size) ),
                                 t_mng.register(
                    ActivationTensor(tensor_size=o_mng.operators[_op_uid].input_b.tensor_size) ),
                                 t_mng.register(
                    ActivationTensor(tensor_size=o_mng.operators[_op_uid].input_c.tensor_size) ),]
            _flatten_op.output = [t_mng.register(
                    ActivationTensor(tensor_size=_output.tensor_size) ) 
                for _output in o_mng.operators[_op_uid].output ]
            _flatten_op.weight = None if o_mng.operators[_op_uid].weight is None else t_mng.register(
                    WeightTensor(tensor_size=o_mng.operators[_op_uid].weight.tensor_size) )
            _flatten_op.intermediate_memory = o_mng.operators[_op_uid].intermediate_memory
            _flatten_op.estimate_runtime = o_mng.operators[_op_uid].estimate_runtime
            _flatten_op.device_name = o_mng.operators[_op_uid].device_name.copy()
            _flatten_op.callback = None
        elif isinstance(o_mng.operators[_op_uid], InitiateOperator):
            _flatten_op.inputs = [t_mng.register(
                    ActivationTensor(tensor_size=_input.tensor_size) ) 
                for _input in o_mng.operators[_op_uid].inputs ]
            _flatten_op.weight = [t_mng.register(
                    WeightTensor(tensor_size=_weight.tensor_size) ) 
                for _weight in o_mng.operators[_op_uid].weight ]
            _flatten_op.estimate_runtime = o_mng.operators[_op_uid].estimate_runtime
            _flatten_op.device_name = (o_mng.operators[_op_uid].device_name+'.')[:-1]
            _flatten_op.callback = None
        else:
            raise Exception("Unknown operator type")
    return _flatten_op_seq_ret