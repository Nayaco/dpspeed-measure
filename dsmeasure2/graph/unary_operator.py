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

def make_linear(input_t: AbstractTensor, output_t: AbstractTensor, weight_t: AbstractTensor, 
                rt_fwd: int = 0, rt_bwd: int = 0) -> tuple[UnaryOperator, BinaryOperator]:

    grad_in_t = TensorManager().register(ActivationTensor(tensor_size=input_t.tensor_size))
    
    linear_forward = UnaryOperator(OperatorComputationalConfig(op_name="linear_forward",))
    linear_backward = BinaryOperator(OperatorComputationalConfig(op_name="linear_backward",))
    
    linear_forward.weight = weight_t
    linear_forward.input = input_t
    linear_forward.output = [output_t]
    linear_forward.estimate_runtime = rt_fwd

    linear_backward.weight = weight_t
    linear_backward.input_a = input_t
    linear_backward.output = [grad_in_t]
    linear_backward.estimate_runtime = rt_bwd

    return linear_forward, linear_backward

def make_gelu(input_t: AbstractTensor, output_t: AbstractTensor, 
              rt_fwd: int = 0, rt_bwd: int = 0) -> tuple[UnaryOperator, BinaryOperator]:

    grad_in_t = TensorManager().register(ActivationTensor(tensor_size=input_t.tensor_size))
    
    gelu_forward = UnaryOperator(OperatorComputationalConfig(op_name="gelu_forward",))
    gelu_backward = BinaryOperator(OperatorComputationalConfig(op_name="gelu_backward",))
    
    gelu_forward.input = input_t
    gelu_forward.output = [output_t]
    gelu_forward.estimate_runtime = rt_fwd
    
    gelu_backward.input_a = input_t
    gelu_backward.output = [grad_in_t]
    gelu_backward.estimate_runtime = rt_bwd

    return gelu_forward, gelu_backward

def make_layernorm(input_t: AbstractTensor, output_t: AbstractTensor, 
                   rt_fwd: int = 0, rt_bwd: int = 0) -> tuple[UnaryOperator, BinaryOperator]:

    grad_in_t = TensorManager().register(ActivationTensor(tensor_size=input_t.tensor_size))
    
    layernorm_forward = UnaryOperator(OperatorComputationalConfig(op_name="layernorm_forward",))
    layernorm_backward = BinaryOperator(OperatorComputationalConfig(op_name="layernorm_backward",))
    
    layernorm_forward.input = input_t
    layernorm_forward.output = [output_t]
    layernorm_forward.estimate_runtime = rt_fwd

    layernorm_backward.input_a = input_t
    layernorm_backward.output = [grad_in_t]
    layernorm_backward.estimate_runtime = rt_bwd

    return layernorm_forward, layernorm_backward

def make_dropout(input_t: AbstractTensor, output_t: AbstractTensor, 
                 rt_fwd: int = 0, rt_bwd: int = 0) -> tuple[UnaryOperator, TernaryOperator]:

    grad_in_t = TensorManager().register(ActivationTensor(tensor_size=input_t.tensor_size))
    mask_t = TensorManager().register(ActivationTensor(tensor_size=int(output_t.tensor_size / 2)))
    
    dropout_forward = UnaryOperator(OperatorComputationalConfig(op_name="dropout_forward",))
    dropout_backward = TernaryOperator(OperatorComputationalConfig(op_name="dropout_backward",))
    
    dropout_forward.input = input_t
    dropout_forward.output = [output_t, mask_t]
    dropout_forward.estimate_runtime = rt_fwd
    
    dropout_backward.input_a = output_t
    dropout_backward.input_b = mask_t
    dropout_backward.output = [grad_in_t]
    dropout_backward.estimate_runtime = rt_bwd

    return dropout_forward, dropout_backward

def make_softmax(input_t: AbstractTensor, output_t: AbstractTensor, 
                 rt_fwd: int = 0, rt_bwd: int = 0) -> tuple[UnaryOperator, BinaryOperator]:

    grad_in_t = TensorManager().register(ActivationTensor(tensor_size=input_t.tensor_size))
    
    softmax_forward = UnaryOperator(OperatorComputationalConfig(op_name="softmax_forward",))
    softmax_backward = BinaryOperator(OperatorComputationalConfig(op_name="softmax_backward",))
    
    softmax_forward.input = input_t
    softmax_forward.output = [output_t]
    softmax_forward.estimate_runtime = rt_fwd
    
    softmax_backward.input_a = output_t
    softmax_backward.output = [grad_in_t]
    softmax_backward.estimate_runtime = rt_bwd

    return softmax_forward, softmax_backward

if __name__ == "__main__":
    input_t_1 = TensorManager().register(ActivationTensor(tensor_size=1024))
    output_t_1 = TensorManager().register(ActivationTensor(tensor_size=1024))
    weight_t_1 = TensorManager().register(ActivationTensor(tensor_size=1024))
    linear_fwd, linear_bwd = make_linear(input_t_1, output_t_1, weight_t_1)
    print(linear_fwd.input)
    print(linear_fwd.weight)
    print([str(i) for i in linear_fwd.output])

    print(linear_bwd.input_a, linear_bwd.input_b)
    print(linear_bwd.weight)
    print([str(i) for i in linear_bwd.output])

    print("=====")

    input_t_2 = TensorManager().register(ActivationTensor(tensor_size=1024))
    output_t_2 = TensorManager().register(ActivationTensor(tensor_size=1024))
    dropout_fwd, dropout_bwd = make_dropout(input_t_2, output_t_2)
    print(dropout_fwd.input)
    print(dropout_fwd.weight)
    print([str(i) for i in dropout_fwd.output])

    print(dropout_bwd.input_a, dropout_bwd.input_b)
    print(dropout_bwd.weight)
    print([str(i) for i in dropout_bwd.output])