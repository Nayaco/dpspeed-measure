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

from dsmeasure2.core.dsm_device_mng   import DeviceManager
from dsmeasure2.core.dsm_operator_mng import OperatorManager

from dsmeasure2.core.dsm_tensor   import AbstractTensor
from dsmeasure2.core.dsm_device   import AbstractDeviceConfig, AbstractDevice
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

from dsmeasure2.core.dsm_tensor_mng    import TensorManager
from dsmeasure2.graph.tensor_define    import ActivationTensor, WeightTensor, TensorState
from dsmeasure2.graph.operator_graph   import UnaryOperator, BinaryOperator, TernaryOperator, InitiateOperator
from dsmeasure2.graph.unary_operator   import make_linear, make_layernorm, make_dropout, make_gelu, make_softmax
from dsmeasure2.graph.binary_operator  import make_add, make_matmul
from dsmeasure2.graph.operator_attn    import make_attn_tp, AttentionTPCRParallel, AttentionTPCRParallelBackward
from dsmeasure2.graph.dsm2_transformer import make_ffn_gpt2, FeedForwardGPT2, FeedForwardGPT2Backward, \
                                              make_transformer_block, TransformerBlockGPT2, TransformerBlockGPT2Backward

def make_gpt_2(
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

    compute_time_loss_with_backward: int,

    batch_size: int,
    seq_len: int,
    hidden_size: int,
    head_num: int,
    head_hidden_size: int,
    tensor_parallel: int|None = None,
    precision: int = 2,

    transfomer_block_num: int = 1  
) -> list[AbstractOperator]:
    _BSH = batch_size*seq_len*hidden_size
    _mb = 1024 * 1024
    input_tensor = TensorManager().register(ActivationTensor(
        tensor_size=precision*_BSH/_mb
    ))

    transfomer_blocks: list[TransformerBlockGPT2] = [None]*transfomer_block_num
    transfomer_blocks_backward: list[TransformerBlockGPT2Backward] = [None]*transfomer_block_num
    for i in range(transfomer_block_num):
        transfomer_blocks[i], transfomer_blocks_backward[i] = make_transformer_block(
            input_tensor if i == 0 else transfomer_blocks[i-1].get_output(),

            compute_time_linear_qkv,
            compute_time_matmul_kq, 
            compute_time_sm,
            compute_time_attention_dropout,
            compute_time_matmul_v,
            compute_time_linear,
            compute_time_dropout_attn,

            compute_time_linear_qkv_backward,
            compute_time_matmul_kq_backward,
            compute_time_sm_backward,
            compute_time_attention_dropout_backward,
            compute_time_matmul_v_backward,
            compute_time_linear_backward,
            compute_time_dropout_attn_backward,

            compute_time_allreduce_attn,

            compute_time_linear_1,
            compute_time_gelu,
            compute_time_linear_2,
            compute_time_dropout_ffn,
            
            compute_time_linear_1_backward,
            compute_time_gelu_backward,
            compute_time_linear_2_backward,
            compute_time_dropout_ffn_backward,

            compute_time_allreduce_ffn,
            
            compute_time_layernorm_1,
            compute_time_layernorm_2,
            compute_time_residual_add_1,
            compute_time_residual_add_2,

            compute_time_layernorm_1_backward,
            compute_time_layernorm_2_backward,

            batch_size,
            seq_len,
            hidden_size,
            head_num,
            head_hidden_size,
            tensor_parallel,
            precision,
        )
    loss_fn = UnaryOperator(OperatorComputationalConfig(op_name="loss_fn",))
    loss_fn.estimate_runtime = compute_time_loss_with_backward
    loss_fn.input = transfomer_blocks[-1].get_output() if transfomer_block_num > 0 else input_tensor
    loss_fn.output = [TensorManager().register(ActivationTensor(
                tensor_size=int(precision*_BSH/_mb)
            ))
        ]
    if transfomer_block_num > 0:
        loss_fn.add_next(transfomer_blocks_backward[-1])
    for i in range(transfomer_block_num):
        transfomer_blocks[i].add_next(
                transfomer_blocks[i+1] if i<transfomer_block_num-1 else loss_fn
            )
        if i < transfomer_block_num-1:
            transfomer_blocks_backward[i+1].add_next(transfomer_blocks_backward[i])
        transfomer_blocks_backward[i].set_grad_out_input(
                loss_fn.output[0] if i == transfomer_block_num-1 \
                    else transfomer_blocks_backward[i+1].get_output()
            )
    init_fn = InitiateOperator(OperatorComputationalConfig(op_name="init_fn",))        
    init_fn.add_next(transfomer_blocks[0] if transfomer_block_num > 0 else loss_fn)
    init_fn.inputs = [input_tensor]
    init_fn.weight = []
    for i in range(transfomer_block_num):
        init_fn.weight.extend(transfomer_blocks[i].weights())
    
    return [init_fn, *transfomer_blocks, loss_fn, *transfomer_blocks_backward[::-1]]

if __name__ == "__main__":
    gpt2 = make_gpt_2(
        compute_time_linear_qkv=490,
        compute_time_matmul_kq=214, 
        compute_time_sm=163,
        compute_time_attention_dropout=286,
        compute_time_matmul_v=191,
        compute_time_linear=146,
        compute_time_dropout_attn=140,

        compute_time_linear_qkv_backward=340,
        compute_time_matmul_kq_backward=531,
        compute_time_sm_backward=212,
        compute_time_attention_dropout_backward=248,
        compute_time_matmul_v_backward=360,
        compute_time_linear_backward=250,
        compute_time_dropout_attn_backward=155,

        compute_time_allreduce_attn=2200,

        compute_time_linear_1=512,
        compute_time_gelu=428,
        compute_time_linear_2=483,
        compute_time_dropout_ffn=75,
        
        compute_time_linear_1_backward=442,
        compute_time_gelu_backward=210,
        compute_time_linear_2_backward=100,
        compute_time_dropout_ffn_backward=102,

        compute_time_allreduce_ffn=2200,
        
        compute_time_layernorm_1=67,
        compute_time_layernorm_2=67,
        compute_time_residual_add_1=67,
        compute_time_residual_add_2=67,

        compute_time_layernorm_1_backward=236,
        compute_time_layernorm_2_backward=236,

        compute_time_loss_with_backward=8000,

        batch_size=8,
        seq_len=1024,
        hidden_size=1792,
        head_num=16,
        head_hidden_size=112,
        tensor_parallel=1,
        precision=2,

        transfomer_block_num=1  
    )
    for _op in gpt2:
        OperatorManager().register(_op)
    def print_ops(op_uid: int):
        op_queue: list[int] = [op_uid]
        while len(op_queue) > 0:
            if not OperatorManager().find(op_queue[0])._config.is_prime:
                print('$', OperatorManager().find(op_queue[0]))
                print_ops(OperatorManager().find(op_queue[0]).subop()[0]._config.op_uid)
            else:
                _op = OperatorManager().find(op_queue[0])
                if isinstance(_op, UnaryOperator):
                    print('_1', _op, 
                        [str(__op) for __op in _op._next], 
                        _op.input, 
                        [str(_t) for _t in _op.output])
                elif isinstance(_op, BinaryOperator):
                    print('_2', _op, 
                        [str(__op) for __op in _op._next], 
                        _op.input_a, _op.input_b, 
                        [str(_t) for _t in _op.output])
                elif isinstance(_op, TernaryOperator):
                    print('_3', _op, 
                        [str(__op) for __op in _op._next], 
                        _op.input_a, _op.input_b, _op.input_c, 
                        [str(_t) for _t in _op.output])
            op_queue.extend([_op._config.op_uid for _op in OperatorManager().find(op_queue[0])._next])
            op_queue.pop(0)
    print_ops(gpt2[0]._config.op_uid)

