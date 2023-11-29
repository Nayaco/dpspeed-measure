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
from typing import Tuple

import warnings

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
from torch import Tensor

from dsmeasure.core.abstract_operator import AbstractOperatorConfig, AbstractOperator
from dsmeasure.common_operators.op_common import OpStaticComputational, OpStaticNonComputational
from dsmeasure.common_operators.op_config import OperatorComputationalConfig, OperatorNonComputationalConfig, OperatorCustomConfig
from dsmeasure.device.gpu import DeviceCUDAConfig, DeviceCUDA

from dsmeasure.core.device_manager import DeviceManager
from dsmeasure.core.operator_manager import OperatorManager

from dsmeasure.core.abstract_tensor import AbstractTensor


class CudaMalloc(OpStaticNonComputational):
    def __init__(self, config: OperatorNonComputationalConfig, alloc_memory: int):
        """
        config: OperatorComputationalConfig
        alloc_memory: int
        """
        super().__init__(config)
        self.estimate_runtime: int = int(1)
        self.alloc_memory = alloc_memory

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)

    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        if cuda is None:
            raise Exception("device not found")
        return cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                           memory=int(self.alloc_memory / 1024 / 1024), computational=False)

class CommonComputational(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, compute_time: int, memory: int = 0):
        super().__init__(config)
        self.estimate_runtime: int = compute_time
        self.memory = memory

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)

    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        if cuda is None:
            raise Exception("device not found")
        return cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                           memory=int(self.memory / 1024 / 1024), computational=True)

class AllReduceOperator(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, compute_time: int, memory: int = 0):
        super().__init__(config)
        self.estimate_runtime: int = compute_time
        self.memory = memory

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)

    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        pcie: DeviceCUDA = DeviceManager().find_by_name('pcie:0')

        if cuda is None or pcie is None:
            raise Exception("device not found")
        could_do = cuda.try_occupy(self.estimate_runtime, memory=0, computational=True) and \
            pcie.try_occupy(self.estimate_runtime)
        return  could_do and \
                cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                            memory=int(self.memory / 1024 / 1024), computational=True) and \
                pcie.occupy(self.estimate_runtime, self.default_apply_cb, \
                            dsize=0)

class AllReduceLayer(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 middle_size: int = 0):
        super().__init__(config)
        # computation
        self.ar_op = AllReduceOperator(OperatorComputationalConfig(
            op_name="ar_op_%s" % config.op_name), 
            compute_time, 
            middle_size)
        self.ar_op_backward = AllReduceOperator(OperatorComputationalConfig(
            op_name="ar_op_backward_%s" % config.op_name), 
            compute_time,
            middle_size)
        
        self._subop = self.ar_op

    def add_next(self, next_op, slot=0):
        self.ar_op.add_next(next_op)
        if isinstance(next_op, UnaryLayer):
            # next op backward -> op gradin / op output free -> 
            # op backward -> next op gradin free
            next_op.unary_op_backward.add_next(self.ar_op_backward)
            self.ar_op_backward.add_next(next_op.gradin_free)
        elif isinstance(next_op, BinaryLayer):
            # next op backward -> op gradin / op output free -> 
            # op backward -> next op gradin free(slot)
            next_op.binary_op_backward.add_next(self.ar_op_backward)
            self.ar_op_backward.add_next(
                next_op.gradin_free_a if slot == 0 else next_op.gradin_free_b)

class UnaryLayer(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 compute_time_backward: int,
                 input_size: int,
                 output_size: int,
                 middle_size: int = 0):
        super().__init__(config)
        # output tensor / gradin tensor
        self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc_%s_%d" % (config.op_name, output_size)), output_size)
        self.gradin_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="gradin_malloc_%s_%d" % (config.op_name, input_size)), input_size)
        # computation
        self.unary_op = CommonComputational(OperatorComputationalConfig(
            op_name="unary_op_%s" % config.op_name), 
            compute_time, 
            middle_size)
        self.unary_op_backward = CommonComputational(OperatorComputationalConfig(
            op_name="unary_op_backward_%s" % config.op_name), 
            compute_time_backward,
            middle_size)
        # free output tensor / gradin tensor
        self.output_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_free_%s_%d" % (config.op_name, -output_size)), -output_size)
        self.gradin_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="gradin_free_%s_%d" % (config.op_name, -input_size)), -input_size)
        
        self.output_malloc.add_next(self.unary_op)
        self.gradin_malloc.add_next(self.unary_op_backward)

        self._subop = self.output_malloc

    def add_next(self, next_op, slot=0):
        self.unary_op.add_next(next_op)
        if isinstance(next_op, UnaryLayer):
            # next op backward -> op gradin / op output free -> 
            # op backward -> next op gradin free
            next_op.unary_op_backward.add_next(self.gradin_malloc)
            next_op.unary_op_backward.add_next(self.output_free)
            self.unary_op_backward.add_next(next_op.gradin_free)
        elif isinstance(next_op, BinaryLayer):
            # next op backward -> op gradin / op output free -> 
            # op backward -> next op gradin free(slot)
            next_op.binary_op_backward.add_next(self.gradin_malloc)
            next_op.binary_op_backward.add_next(self.output_free)
            self.unary_op_backward.add_next(
                next_op.gradin_free_a if slot == 0 else next_op.gradin_free_b)
        elif isinstance(next_op, AllReduceLayer):
            # next op backward -> op gradin / op output free -> 
            # op backward -> next op gradin free
            next_op.ar_op_backward.add_next(self.gradin_malloc)
            next_op.ar_op_backward.add_next(self.output_free)

class BinaryLayer(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 compute_time_backward: int,
                 input_size_a: int,
                 input_size_b: int,
                 output_size: int):
        super().__init__(config)
        # output tensor / gradin tensor
        self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc_%s_%d" % (config.op_name, output_size)), output_size)
        self.gradin_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="gradin_malloc_%s_%d" % (config.op_name, input_size_a + input_size_b)), input_size_a + input_size_b)
        # computation
        self.binary_op = CommonComputational(OperatorComputationalConfig(
            op_name="binary_op_%s" % config.op_name), compute_time)
        self.binary_op_backward = CommonComputational(OperatorComputationalConfig(
            op_name="binary_op_backward_%s" % config.op_name), compute_time_backward)
        # free output tensor / gradin tensor
        self.output_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_free_%s_%d" % (config.op_name, -output_size)), -output_size)
        self.gradin_free_a = CudaMalloc(OperatorNonComputationalConfig(
            op_name="gradin_free_a_%s_%d" % (config.op_name, -input_size_a)), -input_size_a)
        self.gradin_free_b = CudaMalloc(OperatorNonComputationalConfig(
            op_name="gradin_free_b_%s_%d" % (config.op_name, -input_size_b)), -input_size_b)
        
        self.output_malloc.add_next(self.binary_op)
        self.gradin_malloc.add_next(self.binary_op_backward)

        self._subop = self.output_malloc

    def add_next(self, next_op, slot=0):
        self.binary_op.add_next(next_op)
        if isinstance(next_op, UnaryLayer):
            # next op backward -> op gradin / op output free -> 
            # op backward -> next op gradin free
            next_op.unary_op_backward.add_next(self.gradin_malloc)
            next_op.unary_op_backward.add_next(self.output_free)
            self.binary_op_backward.add_next(next_op.gradin_free)
        elif isinstance(next_op, BinaryLayer):
            # next op backward -> op gradin / op output free -> 
            # op backward -> next op gradin free(slot)
            next_op.binary_op_backward.add_next(self.gradin_malloc)
            next_op.binary_op_backward.add_next(self.output_free)
            self.binary_op_backward.add_next(
                next_op.gradin_free_a if slot == 0 else next_op.gradin_free_b)
        elif isinstance(next_op, AllReduceLayer):
            # next op backward -> op gradin / op output free -> 
            # op backward -> next op gradin free
            next_op.ar_op_backward.add_next(self.gradin_malloc)
            next_op.ar_op_backward.add_next(self.output_free)

class CoreAttentionTPCRParallel(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig,

                 compute_time_linear_qkv: int,
                 compute_time_matmul_kq: int, 
                 compute_time_sm: int,
                 compute_time_attention_dropout: int,
                 compute_time_matmul_v: int,
                 compute_time_linear: int,
                 compute_time_allreduce: int,
                 compute_time_dropout: int,

                 compute_time_linear_qkv_backward: int,
                 compute_time_matmul_kq_backward: int,
                 compute_time_sm_backward: int,
                 compute_time_attention_dropout_backward: int,
                 compute_time_matmul_v_backward: int,
                 compute_time_linear_backward: int,
                 compute_time_dropout_backward: int,

                 batch_size: int,
                 seq_len: int,
                 head_num: int,
                 head_hidden_size: int,
                 tensor_parallel: int|None = None,
                 precision: int = 2):
        super().__init__(config)
        tensor_parallel = 1 if tensor_parallel is None else tensor_parallel
        # (B x S x N x H) => 3 x (B x S x N x H)
        self.linear_qkv = UnaryLayer(OperatorCustomConfig(
            op_name="attention_linear_qkv"),
            compute_time_linear_qkv,
            compute_time_linear_qkv_backward,
            precision * int(batch_size*seq_len*head_num*head_hidden_size),
            precision * int(3 * (head_num*head_hidden_size) ** 2) /  tensor_parallel)
        # 2 x (B x S x N x H) => B x S x N x S
        self.matmul_kq = BinaryLayer(OperatorCustomConfig(
            op_name="attention_matmul_kq"), 
            compute_time_matmul_kq, 
            compute_time_matmul_kq_backward,
            precision * int(batch_size*seq_len*head_num*head_hidden_size / tensor_parallel),
            precision * int(batch_size*seq_len*head_num*head_hidden_size / tensor_parallel),
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel))
        # B x S x N x S => B x S x N x S
        self.softmax = UnaryLayer(OperatorCustomConfig(
            op_name="attention_softmax"),
            compute_time_sm, 
            compute_time_sm_backward,
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel),
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel))
        # B x S x N x S => B x S x N x S + [mask(1byte) B x S x N x S]
        self.attention_dropout = UnaryLayer(OperatorCustomConfig(
            op_name="attention_dropout"),
            compute_time_attention_dropout,
            compute_time_attention_dropout_backward,
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel),
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel) + \
                int(batch_size*seq_len*head_num*seq_len / tensor_parallel))
        # B x S x N x S @ B x S x N x H => B x S x N x H
        self.matmul_v = BinaryLayer(OperatorCustomConfig(
            op_name="attention_matmul_v"),
            compute_time_matmul_v,
            compute_time_matmul_v_backward,
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel),
            precision * int(batch_size*seq_len*head_num*head_hidden_size / tensor_parallel),
            precision * int(batch_size*seq_len*head_num*head_hidden_size / tensor_parallel))
        # B x S x N x H => [B x S x N x H]
        self.attention_linear = UnaryLayer(OperatorCustomConfig(
            op_name="attention_linear"),
            compute_time_linear,
            compute_time_linear_backward,
            precision * int(batch_size*seq_len*head_num*head_hidden_size) / tensor_parallel,
            0,
            precision * int(batch_size*seq_len*head_num*head_hidden_size) / tensor_parallel)
        # all reducing ...
        self.attention_allreduce = AllReduceLayer(OperatorCustomConfig(
            op_name="attention_allreduce"),
            compute_time_allreduce,
            precision * int(batch_size*seq_len*head_num*head_hidden_size))
        # B x S x N x H => B x S x N x H
        self.output_dropout = UnaryLayer(OperatorCustomConfig(
            op_name="output_dropout"),
            compute_time_dropout,
            compute_time_dropout_backward,
            0,
            precision * int(batch_size*seq_len*head_num*head_hidden_size),
            precision * int(batch_size*seq_len*head_num*head_hidden_size) + \
                int(batch_size*seq_len*head_num*head_hidden_size))

        self.linear_qkv.add_next(self.matmul_kq, 0)
        self.linear_qkv.add_next(self.matmul_kq, 1)
        self.linear_qkv.add_next(self.matmul_v, 0)

        self.matmul_kq.add_next(self.softmax)
        self.softmax.add_next(self.attention_dropout)
        self.attention_dropout.add_next(self.matmul_v, 1)
        self.matmul_v.add_next(self.attention_linear)

        self.attention_linear.add_next(self.attention_allreduce)
        self.attention_allreduce.add_next(self.output_dropout)

        self._subop = self.linear_qkv

    def add_next(self, next_op):
        self.output_dropout.add_next(next_op)

class FeedForwardLayerGPT(AbstractOperator):
    def __init__(self,
                 config: OperatorCustomConfig,
                 compute_time_layer1: int,
                 compute_time_gelu : int,
                 compute_time_layer2: int,
                 compute_time_allreduce: int,
                 compute_time_dropout: int,
                 
                 compute_time_layer1_backward: int,
                 compute_time_gelu_backward : int,
                 compute_time_layer2_backward: int,
                 compute_time_dropout_backward: int,
            
                 batch_size: int,
                 seq_len: int,
                 hidden_size: int,
                 tensor_parallel: int|None = None,
                 precision: int = 2):
        super().__init__(config)
        tensor_parallel = 1 if tensor_parallel is None else tensor_parallel
        # (B x S x H) => (B x S x 4H)
        self.ffn_linear_1 = UnaryLayer(OperatorCustomConfig(
            op_name="ffn_linear_1"),
            compute_time_layer1,
            compute_time_layer1_backward,
            precision * int(batch_size*seq_len*hidden_size / tensor_parallel),
            precision * int(4*batch_size*seq_len*hidden_size / tensor_parallel))
        # (B x S x 4H) => (B x S x 4H)
        self.ffn_gelu = UnaryLayer(OperatorCustomConfig(
            op_name="ffn_gelu"),
            compute_time_gelu,
            compute_time_gelu_backward,
            precision * int(4*batch_size*seq_len*hidden_size / tensor_parallel),
            precision * int(4*batch_size*seq_len*hidden_size / tensor_parallel))
        # (B x S x 4H) => (B x S x H)
        self.ffn_linear_2 = UnaryLayer(OperatorCustomConfig(
            op_name="ffn_linear_2"),
            compute_time_layer2,
            compute_time_layer2_backward,
            precision * int(4*batch_size*seq_len*hidden_size / tensor_parallel),
            0,
            precision * int(4*batch_size*seq_len*hidden_size / tensor_parallel))
        # (B x S x H) => (B x S x H)
        # all reducing ...
        self.ffn_allreduce = AllReduceLayer(OperatorCustomConfig(
            op_name="ffn_allreduce"),
            compute_time_allreduce,
            precision * int(batch_size*seq_len*hidden_size))
        # (B x S x H) => (B x S x H)
        self.ffn_dropout = UnaryLayer(OperatorCustomConfig(
            op_name="ffn_dropout"),
            compute_time_dropout,
            compute_time_dropout_backward,
            0,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))
        
        # Linear1 -> Gelu -> Linear2 -> AllReduce -> Dropout
        self.ffn_linear_1.add_next(self.ffn_gelu)
        self.ffn_gelu.add_next(self.ffn_linear_2)
        self.ffn_linear_2.add_next(self.ffn_allreduce)
        self.ffn_linear_2.add_next(self.ffn_allreduce)
        self.ffn_allreduce.add_next(self.ffn_dropout)

        self._subop = self.ffn_linear_1
        
    def add_next(self, next_op):
        self.ffn_dropout.add_next(next_op)

class TransformerBlockGPT(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig,

                 compute_time_att_linear_qkv: int,
                 compute_time_att_matmul_kq: int, 
                 compute_time_att_sm: int,
                 compute_time_att_attention_dropout: int,
                 compute_time_att_matmul_v: int,
                 compute_time_att_linear: int,
                 compute_time_att_allreduce: int,
                 compute_time_att_dropout: int,

                 compute_time_att_linear_qkv_backward: int,
                 compute_time_att_matmul_kq_backward: int,
                 compute_time_att_sm_backward: int,
                 compute_time_att_attention_dropout_backward: int,
                 compute_time_att_matmul_v_backward: int,
                 compute_time_att_linear_backward: int,
                 compute_time_att_dropout_backward: int,
                 
                 compute_time_ffn_layer1: int,
                 compute_time_ffn_gelu : int,
                 compute_time_ffn_layer2: int,
                 compute_time_ffn_allreduce: int,
                 compute_time_ffn_dropout: int,
                 
                 compute_time_ffn_layer1_backward: int,
                 compute_time_ffn_gelu_backward : int,
                 compute_time_ffn_layer2_backward: int,
                 compute_time_ffn_dropout_backward: int,

                 compute_time_layernorm_1: int,
                 compute_time_layernorm_2: int,
                 compute_time_layernorm_1_backward: int,
                 compute_time_layernorm_2_backward: int,
                 
                 compute_time_residual_add_1: int,
                 compute_time_residual_add_2: int,
                 compute_time_residual_add_1_backward: int,
                 compute_time_residual_add_2_backward: int,

                 batch_size: int,
                 seq_len: int,
                 head_num: int,
                 head_hidden_size: int,
                 hidden_size: int,

                 tensor_parallel: int|None = None,
                 precision: int = 2):
        super().__init__(config)
        tensor_parallel = 1 if tensor_parallel is None else tensor_parallel

        self.layernorm_1 = UnaryLayer(OperatorCustomConfig(
            op_name="transformer_layernorm_1"),
            compute_time_layernorm_1,
            compute_time_layernorm_1_backward,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))
        
        self.core_attention = CoreAttentionTPCRParallel(OperatorCustomConfig(
            op_name="transformer_core_attention"),
            compute_time_att_linear_qkv,
            compute_time_att_matmul_kq, 
            compute_time_att_sm,
            compute_time_att_attention_dropout,
            compute_time_att_matmul_v,
            compute_time_att_linear,
            compute_time_att_allreduce,
            compute_time_att_dropout,

            compute_time_att_linear_qkv_backward,
            compute_time_att_matmul_kq_backward,
            compute_time_att_sm_backward,
            compute_time_att_attention_dropout_backward,
            compute_time_att_matmul_v_backward,
            compute_time_att_linear_backward,
            compute_time_att_dropout_backward,

            batch_size,
            seq_len,
            head_num,
            head_hidden_size,
            tensor_parallel,
            precision)

        self.residual_add_1 = BinaryLayer(OperatorCustomConfig(
            op_name="transformer_residual_add_1"),
            compute_time_residual_add_1,
            compute_time_residual_add_1_backward,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))
        
        self.layernorm_2 = UnaryLayer(OperatorCustomConfig(
            op_name="transformer_layernorm_2"),
            compute_time_layernorm_2,
            compute_time_layernorm_2_backward,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))
        
        self.ffn_layer = FeedForwardLayerGPT(OperatorCustomConfig(
            op_name="ffn_layer"),
            compute_time_ffn_layer1,
            compute_time_ffn_gelu,
            compute_time_ffn_layer2,
            compute_time_ffn_allreduce,
            compute_time_ffn_dropout,
            
            compute_time_ffn_layer1_backward,
            compute_time_ffn_gelu_backward,
            compute_time_ffn_layer2_backward,
            compute_time_ffn_dropout_backward,
            
            batch_size,
            seq_len,
            hidden_size,
            tensor_parallel,
            precision)

        self.residual_add_2 = BinaryLayer(OperatorCustomConfig(
            op_name="transformer_residual_add_2"),
            compute_time_residual_add_2,
            compute_time_residual_add_2_backward,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))

        # layernorm_1 -> core_attention -> residual_add_1 -> 
        # layernorm_2 -> ffn_layer -> residual_add_2
        self.layernorm_1.add_next(self.core_attention._subop)
        self.core_attention.output_dropout.add_next(self.residual_add_1, 0)
        self.residual_add_1.add_next(self.layernorm_2)
        self.residual_add_1.add_next(self.residual_add_2, 0)
        self.layernorm_2.add_next(self.ffn_layer._subop)
        self.ffn_layer.ffn_dropout.add_next(self.residual_add_2, 1)

        self._subop = self.layernorm_1

    def add_next(self, next_op):
        self.residual_add_2.add_next(next_op)

class GPT2Model(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig,
                 compute_time_embedding: int,
                 compute_time_embedding_backward: int,
                 
                 compute_time_att_linear_qkv: int,
                 compute_time_att_matmul_kq: int, 
                 compute_time_att_sm: int,
                 compute_time_att_attention_dropout: int,
                 compute_time_att_matmul_v: int,
                 compute_time_att_linear: int,
                 compute_time_att_allreduce: int,
                 compute_time_att_dropout: int,

                 compute_time_att_linear_qkv_backward: int,
                 compute_time_att_matmul_kq_backward: int,
                 compute_time_att_sm_backward: int,
                 compute_time_att_attention_dropout_backward: int,
                 compute_time_att_matmul_v_backward: int,
                 compute_time_att_linear_backward: int,
                 compute_time_att_dropout_backward: int,
                 
                 compute_time_ffn_layer1: int,
                 compute_time_ffn_gelu : int,
                 compute_time_ffn_layer2: int,
                 compute_time_ffn_allreduce: int,
                 compute_time_ffn_dropout: int,
                 
                 compute_time_ffn_layer1_backward: int,
                 compute_time_ffn_gelu_backward : int,
                 compute_time_ffn_layer2_backward: int,
                 compute_time_ffn_dropout_backward: int,

                 compute_time_layernorm_1: int,
                 compute_time_layernorm_2: int,
                 compute_time_layernorm_1_backward: int,
                 compute_time_layernorm_2_backward: int,
                 
                 compute_time_residual_add_1: int,
                 compute_time_residual_add_2: int,
                 compute_time_residual_add_1_backward: int,
                 compute_time_residual_add_2_backward: int,
                 
                 compute_time_layernorm_final: int,
                 compute_time_layernorm_final_backward: int,

                 batch_size: int,
                 seq_len: int,
                 head_num: int,
                 head_hidden_size: int,
                 hidden_size: int,
                 vocab_size: int,
                 tensor_parallel: int|None = None,
                 precision: int = 2,
                 transformer_layers: int = 1):
        super().__init__(config)
        self.transformer_layers = transformer_layers

        tensor_parallel = 1 if tensor_parallel is None else tensor_parallel
        self.embedding = UnaryLayer(OperatorCustomConfig(
            op_name="embedding"),
            compute_time_embedding,
            compute_time_embedding_backward,
            0,
            precision * int(batch_size*seq_len*hidden_size))
        
        self.transformer_blocks = [TransformerBlockGPT(OperatorCustomConfig(
            op_name="transformer_block_%d" % i),
            compute_time_att_linear_qkv,
            compute_time_att_matmul_kq, 
            compute_time_att_sm,
            compute_time_att_attention_dropout,
            compute_time_att_matmul_v,
            compute_time_att_linear,
            compute_time_att_allreduce,
            compute_time_att_dropout,

            compute_time_att_linear_qkv_backward,
            compute_time_att_matmul_kq_backward,
            compute_time_att_sm_backward,
            compute_time_att_attention_dropout_backward,
            compute_time_att_matmul_v_backward,
            compute_time_att_linear_backward,
            compute_time_att_dropout_backward,

            compute_time_ffn_layer1,
            compute_time_ffn_gelu,
            compute_time_ffn_layer2,
            compute_time_ffn_allreduce,
            compute_time_ffn_dropout,
            
            compute_time_ffn_layer1_backward,
            compute_time_ffn_gelu_backward,
            compute_time_ffn_layer2_backward,
            compute_time_ffn_dropout_backward,

            compute_time_layernorm_1,
            compute_time_layernorm_2,
            compute_time_layernorm_1_backward,
            compute_time_layernorm_2_backward,
            
            compute_time_residual_add_1,
            compute_time_residual_add_2,
            compute_time_residual_add_1_backward,
            compute_time_residual_add_2_backward,

            batch_size,
            seq_len,
            head_num,
            head_hidden_size,
            hidden_size,

            tensor_parallel,
            precision) for i in range(transformer_layers)]
        
        self.layernorm_final = UnaryLayer(OperatorCustomConfig(
            op_name="layernorm_final"),
            compute_time_layernorm_final,
            compute_time_layernorm_final_backward,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))
        
        self.loss = UnaryLayer(OperatorCustomConfig(
            op_name="loss"),
            1,
            1,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))
        
        self.embedding.add_next(self.transformer_blocks[0]._subop)
        self.embedding.add_next(self.transformer_blocks[0].residual_add_1, 1)
        for i in range(transformer_layers - 1):
            self.transformer_blocks[i].residual_add_2.add_next(self.transformer_blocks[i+1]._subop)
            self.transformer_blocks[i].residual_add_2.add_next(self.transformer_blocks[i+1].residual_add_1, 1)
        self.transformer_blocks[-1].residual_add_2.add_next(self.layernorm_final)
        self.layernorm_final.add_next(self.loss)
        
        self.loss.unary_op.add_next(self.loss.gradin_malloc)
        # self.loss.unary_op_backward._next = []
        # self.loss.unary_op_backward.add_next(self.loss.output_free)
        
        self._subop = self.embedding

if __name__ == "__main__":
    test_unary_a0 = UnaryLayer(OperatorCustomConfig(op_name="test_unary_a0"), 
                            20, 20, 10, 10)
    test_unary_a1 = UnaryLayer(OperatorCustomConfig(op_name="test_unary_a1"), 
                            20, 20, 10, 10)
    
    test_unary_b = UnaryLayer(OperatorCustomConfig(op_name="test_unary_b"), 
                            20, 20, 10, 10)
    
    test_binary_a = BinaryLayer(OperatorCustomConfig(op_name="test_binary_a"),
                            20, 20, 10, 10, 10)
    test_binary_b = BinaryLayer(OperatorCustomConfig(op_name="test_binary_b"),
                            20, 20, 10, 10, 10)
    
    test_unary_a0.add_next(test_binary_a, 0)
    test_unary_a1.add_next(test_binary_a, 0)
    test_binary_b.add_next(test_unary_b)    
