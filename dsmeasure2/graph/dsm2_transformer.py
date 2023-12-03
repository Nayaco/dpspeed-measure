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
from dsmeasure2.graph.operator_attn import make_attn_tp, AttentionTPCRParallel, AttentionTPCRParallelBackward

class FeedForwardGPT2(OpStaticDerivative):
    def __init__(self, 
                 config: OperatorCustomConfig,
                 linear_1: UnaryOperator,
                 gelu: UnaryOperator,
                 linear_2: UnaryOperator,
                 fwd_allreduce: UnaryOperator,
                 dropout: UnaryOperator):
        super().__init__(config)
        
        linear_1.add_next(gelu)
        gelu.add_next(linear_2)
        linear_2.add_next(fwd_allreduce)
        fwd_allreduce.add_next(dropout)

        dropout.callback = self.default_apply_cb

        self._subop = [linear_1, 
                       gelu, 
                       linear_2, 
                       fwd_allreduce, 
                       dropout]
    
    def weights(self) -> list[WeightTensor]:
        return [self._subop[0].weight, self._subop[2].weight]
    
    def get_output(self):
        """
        output of last dropout layer
        """
        return self._subop[-1].output[0]

class FeedForwardGPT2Backward(OpStaticDerivative):
    def __init__(self, 
                 config: OperatorCustomConfig,
                 dropout_backward: TernaryOperator,
                 linear_2_backward: BinaryOperator,
                 gelu_backward: BinaryOperator,
                 linear_1_backward: BinaryOperator,
                 bwd_allreduce: UnaryOperator):
        super().__init__(config)
        
        dropout_backward.add_next(linear_2_backward)
        linear_2_backward.add_next(gelu_backward)
        gelu_backward.add_next(linear_1_backward)
        linear_1_backward.add_next(bwd_allreduce)
        
        bwd_allreduce.callback = self.default_apply_cb

        self._subop = [dropout_backward,
                       linear_2_backward, 
                       gelu_backward, 
                       linear_1_backward, 
                       bwd_allreduce]
        
    def get_output(self):
        # linear_1_backward.output[0](grad in)
        return self._subop[-2].output[0]
    
    def set_grad_out_input(self, grad_out: ActivationTensor):
        # dropout_backward
        self._subop[0].input_c = grad_out

def make_ffn_gpt2(
    input_x_t: ActivationTensor,
    compute_time_linear_1: int,
    compute_time_gelu : int,
    compute_time_linear_2: int,
    compute_time_dropout: int,
    
    compute_time_linear_1_backward: int,
    compute_time_gelu_backward : int,
    compute_time_linear_2_backward: int,
    compute_time_dropout_backward: int,

    compute_time_allreduce: int,

    batch_size: int,
    seq_len: int,
    hidden_size: int,
    tensor_parallel: int|None = None,
    precision: int = 2 
):
    tensor_parallel = tensor_parallel or 1
    _BSH = batch_size*seq_len*hidden_size
    _mb = 1024 * 1024
    # Linear-1
    linear_1_forward = UnaryOperator(OperatorComputationalConfig(op_name="linear_1_forward",))
    linear_1_forward.estimate_runtime = compute_time_linear_1
    linear_1_forward.input = input_x_t
    linear_1_forward.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*4*_BSH/_mb/tensor_parallel)))
        ]
    linear_1_forward.weight = TensorManager().register(WeightTensor(
            tensor_size=int(precision*4*hidden_size*hidden_size/_mb)))
    # Gelu
    gelu_forward, gelu_backward = make_gelu(
        input_t=linear_1_forward.output[0],
        output_t=TensorManager().register(ActivationTensor(
            tensor_size=int(precision*4*_BSH/_mb/tensor_parallel))),
        rt_fwd=compute_time_gelu,
        rt_bwd=compute_time_gelu_backward
    )
    # Linear-2
    linear_2_forward = UnaryOperator(OperatorComputationalConfig(op_name="linear_2_forward",))
    linear_2_forward.estimate_runtime = compute_time_linear_2
    linear_2_forward.input = gelu_forward.output[0]
    linear_2_forward.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_BSH/_mb)))
        ]
    linear_2_forward.weight = TensorManager().register(WeightTensor(
            tensor_size=int(precision*4*hidden_size*hidden_size/_mb)))
    # AllReduce
    allreduce_forward = UnaryOperator(OperatorComputationalConfig(op_name="allreduce_forward",))
    allreduce_forward.estimate_runtime = compute_time_allreduce
    allreduce_forward.input = linear_2_forward.output[0]
    allreduce_forward.output = []
    # Dropout
    dropout_forward, dropout_backward = make_dropout(
        input_t=linear_2_forward.output[0],
        output_t=TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_BSH/_mb))),
        rt_fwd=compute_time_dropout,
        rt_bwd=compute_time_dropout_backward
    )
    # Dropout Backward
    # dropout_backward
    # Linear-2 Backward
    linear_2_backward = BinaryOperator(OperatorComputationalConfig(op_name="linear_2_backward",))
    linear_2_backward.estimate_runtime = compute_time_linear_2_backward
    linear_2_backward.input_b = dropout_backward.output[0]
    linear_2_backward.input_a = gelu_forward.output[0]
    linear_2_backward.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*4*_BSH/_mb/tensor_parallel)))
        ]
    # Gelu Backward
    gelu_backward.input_b = linear_2_backward.output[0]
    # Linear-1 Backward
    linear_1_backward = BinaryOperator(OperatorComputationalConfig(op_name="linear_1_backward",))
    linear_1_backward.estimate_runtime = compute_time_linear_1_backward
    linear_1_backward.input_b = gelu_backward.output[0]
    linear_1_backward.input_a = input_x_t
    linear_1_backward.output = [TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_BSH/_mb)))
        ]
    # AllReduce Backward
    allreduce_backward = UnaryOperator(OperatorComputationalConfig(op_name="allreduce_backward",))
    allreduce_backward.estimate_runtime = compute_time_allreduce
    allreduce_backward.input = linear_1_backward.output[0]
    allreduce_backward.output = []

    return FeedForwardGPT2(
        config=OperatorCustomConfig(op_name="ffn_gpt2"),
        linear_1=linear_1_forward,
        gelu=gelu_forward,
        linear_2=linear_2_forward,
        fwd_allreduce=allreduce_forward,
        dropout=dropout_forward
    ), FeedForwardGPT2Backward(
        config=OperatorCustomConfig(op_name="ffn_gpt2_backward"),
        dropout_backward=dropout_backward,
        linear_2_backward=linear_2_backward,
        gelu_backward=gelu_backward,
        linear_1_backward=linear_1_backward,
        bwd_allreduce=allreduce_backward
    )

class TransformerBlockGPT2(OpStaticDerivative):
    def __init__(self, 
                 config: OperatorCustomConfig,
                 layernorm_layer_1: UnaryOperator,
                 attn: AttentionTPCRParallel,
                 residual_add_1: BinaryOperator,
                 layernorm_layer_2: UnaryOperator,
                 ffn: FeedForwardGPT2,
                 residual_add_2: BinaryOperator):
        super().__init__(config)
        
        layernorm_layer_1.add_next(attn)
        attn.add_next(residual_add_1)
        residual_add_1.add_next(layernorm_layer_2)
        layernorm_layer_2.add_next(ffn)
        ffn.add_next(residual_add_2)

        residual_add_2.callback = self.default_apply_cb

        self._subop = [layernorm_layer_1, 
                       attn, 
                       residual_add_1, 
                       layernorm_layer_2, 
                       ffn, 
                       residual_add_2]
        
    def weights(self) -> list[WeightTensor]:
        return self._subop[1].weights() + self._subop[4].weights()
    
    def get_output(self):
        """
        output of last dropout layer
        """
        return self._subop[-1].output[0]

class TransformerBlockGPT2Backward(OpStaticDerivative):
    def __init__(self, 
                 config: OperatorCustomConfig,
                 ffn_backward: FeedForwardGPT2Backward,
                 layernorm_layer_2_backward: BinaryOperator,
                 backward_grad_accumulate_1: BinaryOperator,
                 attn_backward: AttentionTPCRParallelBackward,
                 layernorm_layer_1_backward: BinaryOperator,
                 backward_grad_accumulate_2: BinaryOperator
                 ):
        super().__init__(config)
        
        ffn_backward.add_next(layernorm_layer_2_backward)
        layernorm_layer_2_backward.add_next(backward_grad_accumulate_1)
        backward_grad_accumulate_1.add_next(attn_backward)
        attn_backward.add_next(layernorm_layer_1_backward)
        layernorm_layer_1_backward.add_next(backward_grad_accumulate_2)

        backward_grad_accumulate_2.callback = self.default_apply_cb

        self._subop = [ffn_backward, 
                       layernorm_layer_2_backward, 
                       backward_grad_accumulate_1, 
                       attn_backward, 
                       layernorm_layer_1_backward, 
                       backward_grad_accumulate_2]
        
    def get_output(self):
        # layernorm_layer_1_backward.output[0](grad in)
        return self._subop[-2].output[0]
    
    def set_grad_out_input(self, grad_out: ActivationTensor):
        # ffn_backward
        self._subop[0].set_grad_out_input(grad_out)
        # backward_grad_accumulate_1
        self._subop[2].input_b = grad_out

def make_transformer_block(
    input_x_t: ActivationTensor,
    
    compute_time_linear_qkv: int,
    compute_time_matmul_kq: int, 
    compute_time_sm: int,
    compute_time_attention_dropout: int,
    compute_time_matmul_v: int,
    compute_time_linear: int,
    compute_time_dropout_attn: int,

    compute_time_linear_qkv_backward: int,
    compute_time_matmul_kq_backward: int,
    compute_time_sm_backward: int,
    compute_time_attention_dropout_backward: int,
    compute_time_matmul_v_backward: int,
    compute_time_linear_backward: int,
    compute_time_dropout_attn_backward: int,

    compute_time_allreduce_attn: int,

    compute_time_linear_1: int,
    compute_time_gelu : int,
    compute_time_linear_2: int,
    compute_time_dropout_ffn: int,
    
    compute_time_linear_1_backward: int,
    compute_time_gelu_backward : int,
    compute_time_linear_2_backward: int,
    compute_time_dropout_ffn_backward: int,

    compute_time_allreduce_ffn: int,
    
    compute_time_layernorm_1: int,
    compute_time_layernorm_2: int,
    compute_time_residual_add_1: int,
    compute_time_residual_add_2: int,

    compute_time_layernorm_1_backward: int,
    compute_time_layernorm_2_backward: int,

    batch_size: int,
    seq_len: int,
    hidden_size: int,
    head_num: int,
    head_hidden_size: int,
    tensor_parallel: int|None = None,
    precision: int = 2  
):
    _BSH = batch_size*seq_len*hidden_size
    _mb = 1024 * 1024
    # Layernorm-1
    layernorm_1_forward, layernorm_1_backward = make_layernorm(
        input_t=input_x_t,
        output_t=TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_BSH/_mb))),
        rt_fwd=compute_time_layernorm_1,
        rt_bwd=compute_time_layernorm_1_backward
    )
    # Attention
    attn_forward, attn_backward = make_attn_tp(
        input_x_tensor=layernorm_1_forward.output[0],

        compute_time_linear_qkv=compute_time_linear_qkv,
        compute_time_matmul_kq=compute_time_matmul_kq, 
        compute_time_sm=compute_time_sm,
        compute_time_attention_dropout=compute_time_attention_dropout,
        compute_time_matmul_v=compute_time_matmul_v,
        compute_time_linear=compute_time_linear,
        compute_time_dropout=compute_time_dropout_attn,

        compute_time_linear_qkv_backward=compute_time_linear_qkv_backward,
        compute_time_matmul_kq_backward=compute_time_matmul_kq_backward,
        compute_time_sm_backward=compute_time_sm_backward,
        compute_time_attention_dropout_backward=compute_time_attention_dropout_backward,
        compute_time_matmul_v_backward=compute_time_matmul_v_backward,
        compute_time_linear_backward=compute_time_linear_backward,
        compute_time_dropout_backward=compute_time_dropout_attn_backward,

        compute_time_allreduce=compute_time_allreduce_attn,

        batch_size=batch_size,
        seq_len=seq_len,
        head_num=head_num,
        head_hidden_size=head_hidden_size,
        tensor_parallel=tensor_parallel,
        precision=precision
    )
    # Residual-Add-1
    residual_add_1_forward, _ = make_add(
        input_t_a=input_x_t,
        input_t_b=attn_forward.get_output(),
        output_t=TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_BSH/_mb))),
        rt_fwd=compute_time_residual_add_1
    )
    # Layernorm-2
    layernorm_2_forward, layernorm_2_backward = make_layernorm(
        input_t=residual_add_1_forward.output[0],
        output_t=TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_BSH/_mb))),
        rt_fwd=compute_time_layernorm_2,
        rt_bwd=compute_time_layernorm_2_backward
    )
    # FeedForward
    ffn_forward, ffn_backward = make_ffn_gpt2(
        input_x_t=layernorm_2_forward.output[0],
        
        compute_time_linear_1=compute_time_linear_1,
        compute_time_gelu=compute_time_gelu,
        compute_time_linear_2=compute_time_linear_2,
        compute_time_dropout=compute_time_dropout_ffn,
        
        compute_time_linear_1_backward=compute_time_linear_1_backward,
        compute_time_gelu_backward=compute_time_gelu_backward,
        compute_time_linear_2_backward=compute_time_linear_2_backward,
        compute_time_dropout_backward=compute_time_dropout_ffn_backward,

        compute_time_allreduce=compute_time_allreduce_ffn,

        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        tensor_parallel=tensor_parallel,
        precision=precision
    )
    # Residual-Add-2
    residual_add_2_forward, _ = make_add(
        input_t_a=residual_add_1_forward.output[0],
        input_t_b=ffn_forward.get_output(),
        output_t=TensorManager().register(ActivationTensor(
            tensor_size=int(precision*_BSH/_mb))),
        rt_fwd=compute_time_residual_add_2
    )
    # Residual-Add-2 Gradout Pass
    # FeedForward Backward
    ffn_backward.set_grad_out_input(None)  # TransformerBlockGPT2Backward set grad_out
    # Layernorm-2 Backward
    layernorm_2_backward.input_b = ffn_backward.get_output() # grad_out
    # GradAccumulate - Residual-Add-1
    backward_grad_accumulate_res1 = BinaryOperator(
        OperatorComputationalConfig(op_name="backward_grad_accumulate_res1",))
    backward_grad_accumulate_res1.estimate_runtime = 1
    backward_grad_accumulate_res1.input_a = layernorm_2_backward.output[0]
    backward_grad_accumulate_res1.input_b = None # TransformerBlockGPT2Backward set grad_out
    # grad_out inplace accumulate to layernorm_2_backward.output[0]
    backward_grad_accumulate_res1.output = []
    
    # Attention Backward
    attn_backward.set_grad_out_input(layernorm_2_backward.output[0])
    # Layernorm-1 Backward
    layernorm_1_backward.input_b = attn_backward.get_output()
    # GradAccumulate - Input
    backward_grad_accumulate_input = BinaryOperator(
        OperatorComputationalConfig(op_name="backward_grad_accumulate_input",))
    backward_grad_accumulate_input.estimate_runtime = 1
    backward_grad_accumulate_input.input_a = layernorm_1_backward.output[0]
    backward_grad_accumulate_input.input_b = layernorm_2_backward.output[0]
    backward_grad_accumulate_input.output = []

    return TransformerBlockGPT2(
        config=OperatorCustomConfig(op_name="transformer_block_gpt2"),
        layernorm_layer_1=layernorm_1_forward,
        attn=attn_forward,
        residual_add_1=residual_add_1_forward,
        layernorm_layer_2=layernorm_2_forward,
        ffn=ffn_forward,
        residual_add_2=residual_add_2_forward
    ), TransformerBlockGPT2Backward(
        config=OperatorCustomConfig(op_name="transformer_block_gpt2_backward"),
        ffn_backward=ffn_backward,
        layernorm_layer_2_backward=layernorm_2_backward,
        backward_grad_accumulate_1=backward_grad_accumulate_res1,
        attn_backward=attn_backward,
        layernorm_layer_1_backward=layernorm_1_backward,
        backward_grad_accumulate_2=backward_grad_accumulate_input
    )

if __name__ == "__main__":
    input_x = TensorManager().register(ActivationTensor(tensor_size=0))

    grad_out_x = TensorManager().register(ActivationTensor(tensor_size=0))
    # fwd_ffn, bwd_ffn = make_ffn_gpt2(input_x_t=input_x,
    #                 compute_time_linear_1=1,
    #                 compute_time_gelu=1,
    #                 compute_time_linear_2=1,
    #                 compute_time_dropout=1,
    #                 compute_time_linear_1_backward=1,
    #                 compute_time_gelu_backward=1,
    #                 compute_time_linear_2_backward=1,
    #                 compute_time_dropout_backward=1,
    #                 compute_time_allreduce=1,
    #                 batch_size=1,
    #                 seq_len=1,
    #                 hidden_size=1,
    #                 tensor_parallel=1,
    #                 precision=2)
    # print(fwd_ffn)
    # for _op in fwd_ffn._subop:
    #     print(_op, [str(__op) for __op in _op._next])
    # print(bwd_ffn)
    # for _op in bwd_ffn._subop:
    #     print(_op, [str(__op) for __op in _op._next])
    fwd, bwd = make_transformer_block(
    input_x_t = input_x,
    
    compute_time_linear_qkv = 1,
    compute_time_matmul_kq = 1, 
    compute_time_sm = 1,
    compute_time_attention_dropout = 1,
    compute_time_matmul_v = 1,
    compute_time_linear = 1,
    compute_time_dropout_attn = 1,

    compute_time_linear_qkv_backward = 1,
    compute_time_matmul_kq_backward = 1,
    compute_time_sm_backward = 1,
    compute_time_attention_dropout_backward = 1,
    compute_time_matmul_v_backward = 1,
    compute_time_linear_backward = 1,
    compute_time_dropout_attn_backward = 1,

    compute_time_allreduce_attn = 1,

    compute_time_linear_1 = 1,
    compute_time_gelu = 1,
    compute_time_linear_2 = 1,
    compute_time_dropout_ffn = 1,
    
    compute_time_linear_1_backward = 1,
    compute_time_gelu_backward = 1,
    compute_time_linear_2_backward = 1,
    compute_time_dropout_ffn_backward = 1,

    compute_time_allreduce_ffn = 1,
    
    compute_time_layernorm_1 = 1,
    compute_time_layernorm_2 = 1,
    compute_time_residual_add_1 = 1,
    compute_time_residual_add_2 = 1,

    compute_time_layernorm_1_backward = 1,
    compute_time_layernorm_2_backward = 1,

    batch_size = 1,
    seq_len = 1,
    hidden_size = 1,
    head_num = 1,
    head_hidden_size = 1,
    tensor_parallel = 1,
    precision = 1
    )
    bwd.set_grad_out_input(grad_out_x)
    
    print("fwd:", fwd)
    for _op in fwd._subop:
        if isinstance(_op, UnaryOperator):
            print('uni', _op, 
                  [str(__op) for __op in _op._next], 
                  _op.input, 
                  [str(_t) for _t in _op.output])
        elif isinstance(_op, BinaryOperator):
            print('bin', _op, 
                  [str(__op) for __op in _op._next], 
                  _op.input_a, _op.input_b, 
                  [str(_t) for _t in _op.output])
        elif isinstance(_op, TernaryOperator):
            print('tri', _op, 
                  [str(__op) for __op in _op._next], 
                  _op.input_a, _op.input_b, _op.input_c, 
                  [str(_t) for _t in _op.output])
        elif isinstance(_op, AttentionTPCRParallel):
            print('att', _op,
                  [str(__op) for __op in _op._next], 
                   _op._subop[0].input,
                  str(_op.get_output()))
        elif isinstance(_op, FeedForwardGPT2):
            print('ffn', _op,
                  [str(__op) for __op in _op._next], 
                   _op._subop[0].input,
                  str(_op.get_output()))
    print("=======")
    print("bwd:", bwd)
    for _op in bwd._subop:
        if isinstance(_op, UnaryOperator):
            print('uni', _op) 
            print([str(__op) for __op in _op._next], [str(__op) for __op in _op._prev]) 
            print(_op.input, 
                  [str(_t) for _t in _op.output])
        elif isinstance(_op, BinaryOperator):
            print('bin', _op) 
            print([str(__op) for __op in _op._next], [str(__op) for __op in _op._prev]) 
            print(_op.input_a, _op.input_b, 
                  [str(_t) for _t in _op.output])
        elif isinstance(_op, TernaryOperator):
            print('tri', _op) 
            print([str(__op) for __op in _op._next], [str(__op) for __op in _op._prev]) 
            print(_op.input_a, _op.input_b, _op.input_c, 
                  [str(_t) for _t in _op.output])
        elif isinstance(_op, AttentionTPCRParallelBackward):
            print('att', _op)
            print([str(__op) for __op in _op._next], [str(__op) for __op in _op._prev]) 
            print(_op._subop[0].input_a, _op._subop[0].input_b, _op._subop[0].input_c,
                  str(_op.get_output()))
        elif isinstance(_op, FeedForwardGPT2Backward):
            print('ffn', _op)
            print([str(__op) for __op in _op._next], [str(__op) for __op in _op._prev]) 
            print(_op._subop[0].input_a, _op._subop[0].input_b, _op._subop[0].input_c,
                  str(_op.get_output()))
        print()