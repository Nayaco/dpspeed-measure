# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any

from enum import Enum
import math 
import torch
import torch.nn.functional as F
from torch import Tensor

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

from dsmeasure2.core.dsm_device_mng import DeviceManager
from dsmeasure2.device.device_cuda import DeviceCUDA
from dsmeasure2.device.device_pcie import DevicePCIE4

from dsmeasure2.core.dsm_tensor_mng import TensorManager
from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState
from dsmeasure2.graph.operator_graph import UnaryOperator, BinaryOperator, TernaryOperator

def make_add(input_t_a: AbstractTensor, input_t_b: AbstractTensor, output_t: AbstractTensor, 
             rt_fwd: int = 0, rt_bwd: int = 0) -> tuple[BinaryOperator, UnaryOperator]:
    
    assert input_t_a.tensor_size == input_t_b.tensor_size, "Add: tensor size not match"

    grad_in_t = TensorManager().register(ActivationTensor(tensor_size=input_t_a.tensor_size))
    
    add_forward = BinaryOperator(OperatorComputationalConfig(op_name="add_forward",))
    add_backward = UnaryOperator(OperatorComputationalConfig(op_name="add_backward",))
    
    add_forward.input_a = input_t_a
    add_forward.input_b = input_t_b
    add_forward.output = [output_t]
    add_forward.estimate_runtime = rt_fwd
    
    add_backward.output = [grad_in_t]
    add_backward.input = output_t

    return add_forward, add_backward

def make_matmul(input_t_a: AbstractTensor, input_t_b: AbstractTensor, output_t: AbstractTensor, 
                rt_fwd: int = 0, rt_bwd: int = 0) -> tuple[BinaryOperator, TernaryOperator]:

    grad_in_t_a = TensorManager().register(ActivationTensor(tensor_size=input_t_a.tensor_size))
    grad_in_t_b = TensorManager().register(ActivationTensor(tensor_size=input_t_a.tensor_size))
    
    matmul_forward = BinaryOperator(OperatorComputationalConfig(op_name="matmul_forward",))
    matmul_backward = TernaryOperator(OperatorComputationalConfig(op_name="matmul_backward",))
    
    matmul_forward.input_a = input_t_a
    matmul_forward.input_b = input_t_b
    matmul_forward.output = [output_t]
    matmul_forward.estimate_runtime = rt_fwd

    matmul_backward.input_a = input_t_a
    matmul_backward.input_b = input_t_b
    matmul_backward.output = [grad_in_t_a, grad_in_t_b]
    matmul_backward.estimate_runtime = rt_bwd

    return matmul_forward, matmul_backward
