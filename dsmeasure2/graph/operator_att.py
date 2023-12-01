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
from dsmeasure2.graph.unary_operator import make_linear, make_layernorm, make_dropout, make_gelu, make_softmax
from dsmeasure2.graph.binary_operator import make_add, make_matmul

class AttentionTPCRParallel(OpStaticDerivative):
    def __init__(self, 
                 config: OperatorCustomConfig,
                 linear_qkv: UnaryOperator,
                 matmul_qk: BinaryOperator,
                 softmax_qk: UnaryOperator,
                 attention_dropout: UnaryOperator,
                 matmul_v: BinaryOperator,
                 output_linear: UnaryOperator,
                 fwd_allreduce: UnaryOperator,
                 dropout: UnaryOperator):
        super().__init__(config)
        
        linear_qkv.add_next(matmul_qk)
        matmul_qk.add_next(softmax_qk)
        softmax_qk.add_next(attention_dropout)
        attention_dropout.add_next(matmul_v)
        matmul_v.add_next(output_linear)
        output_linear.add_next(fwd_allreduce)
        fwd_allreduce.add_next(dropout)

        dropout.callback = self.default_apply_cb

        self._subop = [linear_qkv, 
                       matmul_qk, 
                       softmax_qk, 
                       attention_dropout, 
                       matmul_v, 
                       output_linear, 
                       fwd_allreduce, 
                       dropout]
        
    def get_output(self):
        """
        output of last dropout layer
        """
        return self._subop[-1].output[0]

class AttentionTPCRParallelBackward(OpStaticDerivative):
    def __init__(self, 
                 config: OperatorCustomConfig,
                 dropout_backward: TernaryOperator,
                 output_linear_backward: BinaryOperator,
                 matmul_v_backward: TernaryOperator,
                 attention_dropout_backward: TernaryOperator,
                 softmax_qk_backward: BinaryOperator,
                 matmul_qk_backward: TernaryOperator,
                 linear_qkv_backward: BinaryOperator,
                 bwd_allreduce: UnaryOperator):
        super().__init__(config)

        dropout_backward.add_next(output_linear_backward)
        output_linear_backward.add_next(matmul_v_backward)
        matmul_v_backward.add_next(attention_dropout_backward)
        attention_dropout_backward.add_next(softmax_qk_backward)
        softmax_qk_backward.add_next(matmul_qk_backward)
        matmul_qk_backward.add_next(linear_qkv_backward)
        linear_qkv_backward.add_next(bwd_allreduce)

        bwd_allreduce.callback = self.default_apply_cb

        self._subop = [dropout_backward, 
                       output_linear_backward, 
                       matmul_v_backward, 
                       attention_dropout_backward, 
                       softmax_qk_backward, 
                       matmul_qk_backward, 
                       linear_qkv_backward, 
                       bwd_allreduce]
    def get_output(self):
        """
        grad_in output of linear qkv backward layer
        """
        return self._subop[-2].output[0]

def make_attention_tp(
        input_x_tensor : ActivationTensor,
        compute_time_linear_qkv: int,
        compute_time_matmul_kq: int, 
        compute_time_sm: int,
        compute_time_attention_dropout: int,
        compute_time_matmul_v: int,
        compute_time_linear: int,
        compute_time_dropout: int,

        compute_time_linear_qkv_backward: int,
        compute_time_matmul_kq_backward: int,
        compute_time_sm_backward: int,
        compute_time_attention_dropout_backward: int,
        compute_time_matmul_v_backward: int,
        compute_time_linear_backward: int,
        compute_time_dropout_backward: int,

        compute_time_allreduce: int,

        batch_size: int,
        seq_len: int,
        head_num: int,
        head_hidden_size: int,
        tensor_parallel: int|None = None,
        precision: int = 2):
    
    tensor_parallel = tensor_parallel if tensor_parallel is not None else 1
    _bshn = batch_size*seq_len*head_num*head_hidden_size
    _bssn = batch_size*head_num*seq_len*seq_len
    _mb = 1024 * 1024
    # Linear QKV    (output tp)
    linear_qkv = UnaryOperator(OperatorComputationalConfig(op_name="linear_qkv",))
    linear_qkv.estimate_runtime = compute_time_linear_qkv
    linear_qkv.input = input_x_tensor
    linear_qkv.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*3*_bshn/_mb/tensor_parallel)))
        ]
    linear_qkv.weight = TensorManager().register(WeightTensor(
                tensor_size=int(precision*3*(head_num*head_hidden_size)**2/_mb)))
    # Q x K^T       (output tp)
    matmul_qk = UnaryOperator(OperatorComputationalConfig(op_name="matmul_qk",))
    matmul_qk.estimate_runtime = compute_time_matmul_kq
    matmul_qk.input = linear_qkv.output[0] # QK is in the same tensor
    matmul_qk.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_bssn/_mb/tensor_parallel)))
        ]
    # Softmax(*)    (output tp)
    softmax_qk, softmax_qk_backward = make_softmax(
            input_t=matmul_qk.output[0],
            output_t=TensorManager().register(ActivationTensor(
                tensor_size=int(precision*_bssn/_mb/tensor_parallel))),
            rt_fwd=compute_time_sm, rt_bwd=compute_time_sm_backward
        )
    # Dropout(*)    (output tp)
    attention_dropout, attention_dropout_backward = make_dropout(
            input_t=softmax_qk.output[0],
            output_t=TensorManager().register(ActivationTensor(
                tensor_size=int(precision*_bssn/_mb/tensor_parallel))),
            rt_fwd=compute_time_attention_dropout, 
            rt_bwd=compute_time_attention_dropout_backward
        )
    # Dropout(*) x V (output tp)
    matmul_v = BinaryOperator(OperatorComputationalConfig(op_name="matmul_v",))
    matmul_v.estimate_runtime = compute_time_matmul_v
    matmul_v.input_a = attention_dropout.output[0]
    matmul_v.input_b = linear_qkv.output[0]
    matmul_v.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_bshn/_mb/tensor_parallel)))
        ]
    # Linear(CoreAttention(*))
    output_linear, output_linear_backward = make_linear(
            input_t=matmul_v.output[0],
            output_t=TensorManager().register(ActivationTensor(
                tensor_size=precision * int(_bshn/_mb))),
            weight_t=TensorManager().register(WeightTensor(
                tensor_size=int(precision*(head_num*head_hidden_size)**2/_mb))),
            rt_fwd=compute_time_linear, 
            rt_bwd=compute_time_linear_backward
        )
    # Forward Allreduce
    allreduce_forward = UnaryOperator(OperatorComputationalConfig(op_name="allreduce_forward",))
    allreduce_forward.estimate_runtime = compute_time_allreduce
    allreduce_forward.input = output_linear.output[0]
    allreduce_forward.output = []
    # Output Dropout(*)
    output_dropout = UnaryOperator(OperatorComputationalConfig(op_name="output_dropout",))
    output_dropout.estimate_runtime = compute_time_dropout
    output_dropout.input = output_linear.output[0]
    output_dropout.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*1.5*_bshn/_mb)))
        ]
    # Backward Output Dropout(*)
    output_dropout_backward = BinaryOperator(OperatorComputationalConfig(op_name="output_dropout_backward",))
    output_dropout_backward.estimate_runtime = compute_time_dropout_backward
    output_dropout_backward.input_a = output_dropout.output[0]
    output_dropout_backward.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_bshn/_mb)))
        ]
    # Backward Linear(CoreAttention(*)) (output tp)
    output_linear_backward.input_b = output_dropout_backward.output[0]
    # Backward Matmul(*) x V            (output tp)
    matmul_v_backward = TernaryOperator(OperatorComputationalConfig(op_name="matmul_v_backward",))
    matmul_v_backward.estimate_runtime = compute_time_matmul_v_backward
    matmul_v_backward.input_c = output_linear_backward.output[0]
    matmul_v_backward.input_a = attention_dropout.output[0]
    matmul_v_backward.input_b = linear_qkv.output[0]
    matmul_v_backward.output = [
            TensorManager().register(ActivationTensor(
                tensor_size=int(precision*_bshn/_mb/tensor_parallel))),
            TensorManager().register(ActivationTensor(
                tensor_size=int(precision*_bssn/_mb/tensor_parallel)))
        ] # [0]: grad of v, [1]: grad of att_dropout output, 
    # Backward Dropout(*)           (output tp)
    attention_dropout_backward.input_b = matmul_v_backward.output[1]
    # Backward Softmax(*)           (output tp)
    softmax_qk_backward.input_b = attention_dropout_backward.output[0]
    # Backward Q x K^T              (output tp)
    matmul_qk_backward = BinaryOperator(OperatorComputationalConfig(op_name="matmul_qk_backward",))
    matmul_qk_backward.estimate_runtime = compute_time_matmul_kq_backward
    matmul_qk_backward.input_b = softmax_qk_backward.output[0]
    matmul_qk_backward.input_a = linear_qkv.output[0] # QK is in the same tensor
    matmul_qk_backward.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*2*_bshn/_mb/tensor_parallel)))
        ] # so their gradient
    # Backward Linear QKV          (output tp)
    linear_qkv_backward = TernaryOperator(OperatorComputationalConfig(op_name="linear_qkv_backward",))
    linear_qkv_backward.estimate_runtime = compute_time_linear_qkv_backward
    linear_qkv_backward.input_b = matmul_v_backward.output[0]
    linear_qkv_backward.input_c = matmul_qk_backward.output[0]
    linear_qkv_backward.input_a = input_x_tensor
    linear_qkv_backward.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_bshn/_mb)))
        ]
    # Backward Allreduce
    allreduce_backward = UnaryOperator(OperatorComputationalConfig(op_name="allreduce_backward",))
    allreduce_backward.estimate_runtime = compute_time_allreduce
    allreduce_backward.input = linear_qkv_backward.output[0]
    allreduce_backward.output = []

    return AttentionTPCRParallel(
            config=OperatorCustomConfig(op_name="attention_tpcr_parallel",),
            linear_qkv=linear_qkv,
            matmul_qk=matmul_qk,
            softmax_qk=softmax_qk,
            attention_dropout=attention_dropout,
            matmul_v=matmul_v,
            output_linear=output_linear,
            fwd_allreduce=allreduce_forward,
            dropout=output_dropout
        ), AttentionTPCRParallelBackward(
            config=OperatorCustomConfig(op_name="attention_tpcr_parallel_backward",),
            dropout_backward=output_dropout_backward,
            output_linear_backward=output_linear_backward,
            matmul_v_backward=matmul_v_backward,
            attention_dropout_backward=attention_dropout_backward,
            softmax_qk_backward=softmax_qk_backward,
            matmul_qk_backward=matmul_qk_backward,
            linear_qkv_backward=linear_qkv_backward,
            bwd_allreduce=allreduce_backward
        )