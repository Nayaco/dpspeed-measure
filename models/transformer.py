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
                           memory=self.alloc_memory, computational=False)

class CommonComputational(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, compute_time: int):
        super().__init__(config)
        self.estimate_runtime: int = compute_time

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)

    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        if cuda is None:
            raise Exception("device not found")
        return cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                           memory=0, computational=True)

class AllReduceLayer(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, compute_time: int):
        super().__init__(config)
        self.estimate_runtime: int = compute_time

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)

    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        pcie: DeviceCUDA = DeviceManager().find_by_name('pcie:0')

        if cuda or pcie is None:
            raise Exception("device not found")
        could_do = cuda.try_occupy(self.estimate_runtime, memory=0, computational=True) and \
            pcie.try_occupy(self.estimate_runtime)
        return  could_do and \
                cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                            memory=0, computational=True) and \
                pcie.occupy(self.estimate_runtime, self.default_apply_cb, \
                            dsize=0)

class AddLayer(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 output_size: int):
        super().__init__(config)
        # output tensor
        self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc"), output_size)
        # computation
        self.addlayer = CommonComputational(OperatorComputationalConfig(
            op_name="add"), compute_time)
        # free output tensor
        self.output_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_free"), -output_size)
        # Init-Weight -> Output-Malloc -> Add -> Output-Free
        self.output_malloc.add_next(self.addlayer)
        self.addlayer.add_next(self.output_free)

        self._subop = self.output_malloc

    def add_next(self, next_op):
        self.addlayer.add_next(next_op)
    
    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        next_op.add_next(self.output_free)

class AddLayerBackward(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 grad_in_size: int):
        super().__init__(config)
        # grad in tensor
        self.grad_in_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_malloc"), grad_in_size) # 16 bit grad_in
        # backward computation
        self.add_backward = \
            CommonComputational(OperatorComputationalConfig(
                op_name="add_backward"), compute_time)
        # free grad in tensor
        self.grad_in_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_free_a"), -grad_in_size)
        # Grad-In-Malloc -> Add-Backward -> Grad-In-Free
        self.grad_in_malloc.add_next(self.add_backward)
        self.add_backward.add_next(self.grad_in_free)

        self._subop = self.grad_in_malloc

    def add_next(self, next_op):
        raise Exception("AddLayerBackward.add_next() is not supported, use add_next_a() and add_next_b() instead")

    def add_next_a(self, next_op):
        self.add_backward.add_next(next_op)
    
    def add_next_b(self, next_op):
        self.add_backward.add_next(next_op)

    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        next_op.add_next(self.grad_in_free)

class LinearLayerAdam(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 output_size: int, 
                 weight_size: int):
        super().__init__(config)
        # W and b and parameter grad 16 bit
        self.init_weight = CudaMalloc(OperatorNonComputationalConfig(
            op_name="init_weight"), weight_size*2)
        # output tensor
        self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc"), output_size)
        # computation
        self.linear_matmul_bias = CommonComputational(OperatorComputationalConfig(
            op_name="linear_matmul_bias"), compute_time)
        # free output tensor
        self.output_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_free"), -output_size)
        # Init-Weight -> Output-Malloc -> Linear-Matmul-Bias -> Output-Free
        self.init_weight.add_next(self.output_malloc)
        self.output_malloc.add_next(self.linear_matmul_bias)
        self.linear_matmul_bias.add_next(self.output_free)

        self._subop = self.output_malloc

    def add_next(self, next_op):
        self.linear_matmul_bias.add_next(next_op)
        
    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        next_op.add_next(self.output_free)

class LinearLayerBackwardAdam(AbstractOperator):
    def __init__(self, config: OperatorCustomConfig, 
                 compute_time: int, 
                 grad_in_size: int):
        super().__init__(config)
        # grad in tensor
        self.grad_in_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_malloc"), grad_in_size) # 16 bit grad_in
        # backward computation
        self.linear_matmul_bias_backward = \
            CommonComputational(OperatorComputationalConfig(
                op_name="linear_matmul_bias_backward"), compute_time)
        # free grad in tensor
        self.grad_in_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_free"), -grad_in_size)
        # Grad-In-Malloc -> Linear-Matmul-Bias-Backward -> Grad-In-Free
        self.grad_in_malloc.add_next(self.linear_matmul_bias_backward)
        self.linear_matmul_bias_backward.add_next(self.grad_in_free)

        self._subop = self.grad_in_malloc

    def add_next(self, next_op):
        self.linear_matmul_bias_backward.add_next(next_op)

    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        next_op.add_next(self.grad_in_free)

class DropoutLayer(AbstractOperator):
    """
    Dropout-Layer
    output = dropout(input) 
    size(output) == size(input) | size(mask) == size(input) / 2
    """
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int,
                 output_size: int = 0,
                 inplace: bool = False):
        super().__init__(config)
        self.inplace = inplace
        # mask tensor
        self.mask_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="mask_malloc"), int(output_size/2))
        # output tensor
        if not inplace:
            self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc"), output_size)
        # computation
        self.dropout = CommonComputational(OperatorComputationalConfig(
            op_name="dropout"), compute_time)
        # free output tensor
        if not inplace:
            self.output_free = CudaMalloc(OperatorNonComputationalConfig(
                op_name="output_free"), -output_size)
        # free mask tensor
        self.mask_free = CudaMalloc(OperatorNonComputationalConfig(
                op_name="mask_free"), -int(output_size/2))
        if inplace:
            # Mask-Malloc -> Dropout -> Mask-Free
            self.mask_malloc.add_next(self.dropout)
            self.dropout.add_next(self.mask_free)
            self._subop = self.mask_malloc
        else:
            # Mask-Malloc -> Output-Malloc -> Dropout -> Output-Free | Mask-Free
            self.mask_malloc.add_next(self.output_malloc)
            self.output_malloc.add_next(self.dropout)
            self.dropout.add_next(self.output_free)
            self.dropout.add_next(self.mask_free)
            self._subop = self.mask_malloc

    def add_next(self, next_op):
        self.dropout.add_next(next_op)

    def add_next_bf_mask_free(self, next_op):
        """
        append mask tensor free operator to next_op
        """
        next_op.add_next(self.mask_free)

    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        if self.inplace:
            raise Exception("DropoutLayer.add_next_bf_output_free() is not supported when not inplace")
        next_op.add_next(self.output_free)

class DropoutLayerBackward(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int,
                 grad_in_size: int):
        super().__init__(config)
        # grad in tensor
        self.grad_in_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_malloc"), grad_in_size) # 16 bit grad_in
        # backward computation
        self.dropout_backward = \
            CommonComputational(OperatorComputationalConfig(
                op_name="dropout_backward"), compute_time)
        # free grad in tensor
        self.grad_in_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_free"), -grad_in_size)
        # Grad-In-Malloc -> Dropout-Backward -> Grad-In-Free
        self.grad_in_malloc.add_next(self.dropout_backward)
        self.dropout_backward.add_next(self.grad_in_free)

        self._subop = self.grad_in_malloc
    
    def add_next(self, next_op):
        self.dropout_backward.add_next(next_op)
    
    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        next_op.add_next(self.grad_in_free)

class MatMulLayer(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 output_size: int):
        super().__init__(config)
        # output tensor
        self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc"), output_size)
        # computation
        self.matmul = CommonComputational(OperatorComputationalConfig(
            op_name="matmul"), compute_time)
        # free output tensor
        self.output_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_free"), -output_size)
        # Init-Weight -> Output-Malloc -> Matmul -> Output-Free
        self.output_malloc.add_next(self.matmul)
        self.matmul.add_next(self.output_free)

        self._subop = self.output_malloc

    def add_next(self, next_op):
        self.matmul.add_next(next_op)
    
    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        next_op.add_next(self.output_free)

class MatMulLayerBackward(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 grad_in_size_a: int,
                 grad_in_size_b: int):
        super().__init__(config)
        # grad in tensor
        self.grad_in_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_malloc"), grad_in_size_a + grad_in_size_b) # 16 bit grad_in
        # backward computation
        self.matmul_backward = \
            CommonComputational(OperatorComputationalConfig(
                op_name="matmul_backward"), compute_time)
        # free grad in tensor
        self.grad_in_free_a = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_free_a"), -grad_in_size_a)
        self.grad_in_free_b = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_free_b"), -grad_in_size_b)
        # Grad-In-Malloc -> Matmul-Backward -> Grad-In-Free
        self.grad_in_malloc.add_next(self.matmul_backward)
        self.matmul_backward.add_next(self.grad_in_free_a)
        self.matmul_backward.add_next(self.grad_in_free_b)

        self._subop = self.grad_in_malloc

    def add_next(self, next_op):
        raise Exception("MatMulLayerBackward.add_next() is not supported, use add_next_a() and add_next_b() instead")

    def add_next_a(self, next_op):
        self.matmul_backward.add_next(next_op)
    
    def add_next_bf_gradin_free_a(self, next_op):
        """
        append grad in tensor free operator to next_op (A)
        """
        next_op.add_next(self.grad_in_free_a)

    def add_next_b(self, next_op):
        self.matmul_backward.add_next(next_op)

    def add_next_bf_gradin_free_b(self, next_op):
        """
        append grad in tensor free operator to next_op (B)
        """
        next_op.add_next(self.grad_in_free_b)

class SoftmaxLayer(AbstractOperator):
    """
    Softmax-Layer
    output = soft(input) 
    size(output) == size(input)
    """
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int,
                 output_size: int = 0,
                 inplace: bool = False):
        super().__init__(config)
        self.inplace = inplace
        # output tensor
        if not inplace:
            self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc"), output_size)
        # computation
        self.softmax = CommonComputational(OperatorComputationalConfig(
            op_name="softmax"), compute_time)
        # free output tensor
        if not inplace:
            self.output_free = CudaMalloc(OperatorNonComputationalConfig(
                op_name="output_free"), -output_size)
        if inplace:
            # Softmax
            self._subop = self.softmax
        else:
            # Output-Malloc -> Softmax -> Output-Free
            self.output_malloc.add_next(self.softmax)
            self.softmax.add_next(self.output_free)
            self._subop = self.output_malloc

    def add_next(self, next_op):
        self.softmax.add_next(next_op)

    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        if self.inplace:
            raise Exception("SoftmaxLayer.add_next_bf_output_free() is not supported when not inplace")
        next_op.add_next(self.output_free)

class SoftmaxLayerBackward(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int,
                 grad_in_size: int):
        super().__init__(config)
        # grad in tensor
        self.grad_in_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_malloc"), grad_in_size) # 16 bit grad_in
        # backward computation
        self.softmax_backward = \
            CommonComputational(OperatorComputationalConfig(
                op_name="softmax_backward"), compute_time)
        # free grad in tensor
        self.grad_in_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_free"), -grad_in_size)
        # Grad-In-Malloc -> Softmax-Backward -> Grad-In-Free
        self.grad_in_malloc.add_next(self.softmax_backward)
        self.softmax_backward.add_next(self.grad_in_free)

        self._subop = self.grad_in_malloc
    
    def add_next(self, next_op):
        self.softmax_backward.add_next(next_op)
    
    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        next_op.add_next(self.grad_in_free)

class CoreAttentionTPRowParallel(AbstractOperator):
    """
    K --
    Q -- QK^T -> Softmax -- Dropout ->
    V --      --         --         -- Matmul(V) -- Linear -- AllReduce -- Dropout
    """
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
                 compute_time_allreduce_backward: int,
                 compute_time_dropout_backward: int,

                 batch_size: int,
                 seq_len: int,
                 head_num: int,
                 head_hidden_size: int,
                 tensor_parallel: int|None = None,
                 precision: int = 2):
        super().__init__(config)
        tensor_parallel = 1 if tensor_parallel is None else tensor_parallel
        # 3 x (B x S x N x H) => 3 x (B x S x N x H)
        self.linear_qkv = LinearLayerAdam(OperatorCustomConfig(
            op_name="attention_linear_qkv"),
            compute_time_linear_qkv,
            precision * int(3 * batch_size*seq_len*head_num*head_hidden_size / tensor_parallel),
            precision * int(3 * (head_num*head_hidden_size) ** 2) /  tensor_parallel)
        self.linear_qkv_backward = LinearLayerBackwardAdam(OperatorCustomConfig(
            op_name="attention_linear_qkv_backward"),
            compute_time_linear_qkv_backward,
            precision * int(batch_size*seq_len*head_num*head_hidden_size))
        # 2 x (B x S x N x H) => B x S x N x S
        self.matmul_kq = MatMulLayer(OperatorCustomConfig(
            op_name="attention_matmul_kq"), 
            compute_time_matmul_kq, 
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel))
        self.matmul_kq_backward = MatMulLayerBackward(OperatorCustomConfig(
            op_name="attention_matmul_kq_backward"), 
            compute_time_matmul_kq_backward, 
            precision * int(batch_size*seq_len*head_num*head_hidden_size / tensor_parallel),
            precision * int(batch_size*seq_len*head_num*head_hidden_size / tensor_parallel))
        # B x S x N x S => B x S x N x S
        self.softmax = SoftmaxLayer(OperatorCustomConfig(
            op_name="attention_softmax"),
            compute_time_sm, 
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel), 
            inplace=True)
        self.softmax_backward = SoftmaxLayerBackward(OperatorCustomConfig(
            op_name="attention_softmax_backward"),
            compute_time_sm_backward, 
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel))
        # B x S x N x S => B x S x N x S + [mask(1byte) B x S x N x S]
        self.attention_dropout = DropoutLayer(OperatorCustomConfig(
            op_name="attention_dropout"),
            compute_time_attention_dropout,
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel),
            inplace=False)
        self.attention_dropout_backward = DropoutLayerBackward(OperatorCustomConfig(
            op_name="attention_dropout_backward"),
            compute_time_attention_dropout_backward,
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel))
        # B x S x N x S @ B x S x N x H => B x S x N x H
        self.matmul_v = MatMulLayer(OperatorCustomConfig(
            op_name="attention_matmul_v"),
            compute_time_matmul_v,
            precision * int(batch_size*seq_len*head_num*head_hidden_size / tensor_parallel))
        self.matmul_v_backward = MatMulLayerBackward(OperatorCustomConfig(
            op_name="attention_matmul_v_backward"),
            compute_time_matmul_v_backward,
            precision * int(batch_size*seq_len*head_num*seq_len / tensor_parallel),
            precision * int(batch_size*seq_len*head_num*head_hidden_size / tensor_parallel))
        # B x S x N x H => B x S x N x H
        self.attention_linear = LinearLayerAdam(OperatorCustomConfig(
            op_name="attention_linear"),
            compute_time_linear,
            precision * int(batch_size*seq_len*head_num*head_hidden_size),
            precision * int((head_num*head_hidden_size) ** 2))
        self.attention_linear_backward = LinearLayerBackwardAdam(OperatorCustomConfig(
            op_name="attention_linear_backward"),
            compute_time_linear_backward,
            precision * int(batch_size*seq_len*head_num*head_hidden_size))
        # B x S x N x H => B x S x N x H
        self.attention_allreduce = AllReduceLayer(OperatorCustomConfig(
            op_name="attention_allreduce"),
            compute_time_allreduce)
        self.attention_allreduce_backward = AllReduceLayer(OperatorCustomConfig(
            op_name="attention_allreduce_backward"),
            compute_time_allreduce_backward)
        # B x S x N x H => B x S x N x H
        self.output_dropout = DropoutLayer(OperatorCustomConfig(
            op_name="output_dropout"),
            compute_time_dropout,
            precision * int(batch_size*seq_len*head_num*head_hidden_size),
            inplace=True)
        self.output_dropout_backward = DropoutLayerBackward(OperatorCustomConfig(
            op_name="output_dropout_backward"),
            compute_time_dropout_backward,
            precision * int(batch_size*seq_len*head_num*head_hidden_size))
        
        # LinearQKV -> Matmul-KQ -> Softmax -> Dropout -> Matmul-V -> 
        # Linear -> AllReduce -> Dropout
        self.linear_qkv.add_next(self.matmul_kq)
        self.matmul_kq.add_next(self.softmax)
        self.softmax.add_next(self.attention_dropout)
        self.attention_dropout.add_next(self.matmul_v)
        self.matmul_v.add_next(self.attention_linear)
        self.attention_linear.add_next(self.attention_allreduce)
        self.attention_allreduce.add_next(self.output_dropout)
        # MIDLE FORWARD AND BACKWARD
        self.output_dropout.add_next(self.output_dropout_backward)
        # Dropout-Backward -> AllReduce-Backward -> Linear-Backward ->
        # Matmul-V-Backward -> Dropout-Backward -> Softmax-Backward -> 
        # Matmul-KQ-Backward -> Linear-QKV-Backward
        self.output_dropout_backward.add_next(self.attention_allreduce_backward)
        self.attention_allreduce_backward.add_next(self.attention_linear_backward)
        self.attention_linear_backward.add_next(self.matmul_v_backward)
        self.matmul_v_backward.add_next_a(self.attention_dropout_backward)
        self.matmul_v_backward.add_next_b(self.linear_qkv_backward)
        self.attention_dropout_backward.add_next(self.softmax_backward)
        self.softmax_backward.add_next(self.matmul_kq_backward)
        self.matmul_kq_backward.add_next_a(self.linear_qkv_backward)
        self.matmul_kq_backward.add_next_b(self.linear_qkv_backward)
        # linear kqv output
        self.linear_qkv.add_next_bf_output_free(self.matmul_kq_backward)
        # softmax output is matmul-kq output (inplace)
        self.matmul_kq.add_next_bf_output_free(self.softmax_backward)
        # dropout output(NOT inplace)
        self.attention_dropout.add_next_bf_output_free(self.attention_dropout_backward)
        # matmul-v output
        self.matmul_v.add_next_bf_output_free(self.attention_linear_backward)
        # output-dropout output is linear output (inplace)
        self.attention_linear.add_next_bf_output_free(self.output_dropout_backward)
        
        self.output_dropout_backward.add_next_bf_gradin_free(self.attention_linear_backward)
        self.attention_linear_backward.add_next_bf_gradin_free(self.matmul_v_backward)
        self.matmul_v_backward.add_next_bf_gradin_free_a(self.attention_dropout_backward)
        self.matmul_v_backward.add_next_bf_gradin_free_b(self.linear_qkv_backward)
        self.attention_dropout_backward.add_next_bf_gradin_free(self.softmax_backward)
        self.softmax_backward.add_next_bf_gradin_free(self.matmul_kq_backward)
        self.matmul_kq_backward.add_next_bf_gradin_free_a(self.linear_qkv_backward)
        self.matmul_kq_backward.add_next_bf_gradin_free_b(self.linear_qkv_backward)

        self._subop = self.linear_qkv

        self.init_weight = [self.linear_qkv.init_weight, 
                            self.attention_linear.init_weight]


    def add_next(self, next_op):
        self.output_dropout.add_next(next_op)

    def add_next_backward(self, next_backward_op):
        self.linear_qkv_backward.add_next(next_backward_op)
    
    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        self.linear_qkv_backward.add_next_bf_gradin_free(next_op)

class LayerNormLayer(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 output_size: int):
        super().__init__(config)
        # output tensor
        self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc"), output_size)
        # computation
        self.layernorm = CommonComputational(OperatorComputationalConfig(
            op_name="layernorm"), compute_time)
        # free output tensor
        self.output_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_free"), -output_size)
        # Output-Malloc -> LayerNorm -> Output-Free
        self.output_malloc.add_next(self.layernorm)
        self.layernorm.add_next(self.output_free)

        self._subop = self.output_malloc

    def add_next(self, next_op):
        self.layernorm.add_next(next_op)
        
    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        next_op.add_next(self.output_free)

class LayerNormLayerBackward(AbstractOperator):
    def __init__(self, config: OperatorCustomConfig, 
                 compute_time: int, 
                 grad_in_size: int):
        super().__init__(config)
        # grad in tensor
        self.grad_in_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_malloc"), grad_in_size) # 16 bit grad_in
        # backward computation
        self.layernorm_backward = \
            CommonComputational(OperatorComputationalConfig(
                op_name="layernorm_backward"), compute_time)
        # free grad in tensor
        self.grad_in_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_free"), -grad_in_size)
        # Grad-In-Malloc -> LayerNorm-Backward -> Grad-In-Free
        self.grad_in_malloc.add_next(self.layernorm_backward)
        self.layernorm_backward.add_next(self.grad_in_free)

        self._subop = self.grad_in_malloc

    def add_next(self, next_op):
        self.layernorm_backward.add_next(next_op)

    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        next_op.add_next(self.grad_in_free)

class GeluLayer(AbstractOperator):
    """
    Gelu-Layer
    output = soft(input) 
    size(output) == size(input)
    """
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int,
                 output_size: int = 0):
        super().__init__(config)
        # output tensor
        self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
        op_name="output_malloc"), output_size)
        # computation
        self.gelu = CommonComputational(OperatorComputationalConfig(
            op_name="gelu"), compute_time)
        # free output tensor
        self.output_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_free"), -output_size)
        # Output-Malloc -> Gelu -> Output-Free
        self.output_malloc.add_next(self.gelu)
        self.gelu.add_next(self.output_free)
        self._subop = self.output_malloc

    def add_next(self, next_op):
        self.gelu.add_next(next_op)

    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        next_op.add_next(self.output_free)

class GeluLayerBackward(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int,
                 grad_in_size: int):
        super().__init__(config)
        # grad in tensor
        self.grad_in_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_malloc"), grad_in_size) # 16 bit grad_in
        # backward computation
        self.gelu_backward = \
            CommonComputational(OperatorComputationalConfig(
                op_name="gelu_backward"), compute_time)
        # free grad in tensor
        self.grad_in_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="grad_in_free"), -grad_in_size)
        # Grad-In-Malloc -> Gelu-Backward -> Grad-In-Free
        self.grad_in_malloc.add_next(self.gelu_backward)
        self.gelu_backward.add_next(self.grad_in_free)

        self._subop = self.grad_in_malloc
    
    def add_next(self, next_op):
        self.gelu_backward.add_next(next_op)
    
    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        next_op.add_next(self.grad_in_free)


class FeedForwardLayerGPT(AbstractOperator):
    """
    """
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
                 compute_time_allreduce_backward: int,
                 compute_time_dropout_backward: int,
            
                 batch_size: int,
                 seq_len: int,
                 hidden_size: int,
                 tensor_parallel: int|None = None,
                 precision: int = 2):
        super().__init__(config)
        tensor_parallel = 1 if tensor_parallel is None else tensor_parallel
        # (B x S x H) => (B x S x 4H)
        self.ffn_linear_1 = LinearLayerAdam(OperatorCustomConfig(
            op_name="ffn_linear_1"),
            compute_time_layer1,
            precision * int(batch_size*seq_len*4*hidden_size / tensor_parallel),
            precision * int(4 * (hidden_size**2)))
        self.ffn_linear_1_backward = LinearLayerBackwardAdam(OperatorCustomConfig(
            op_name="ffn_linear_1_backward"),
            compute_time_layer1_backward,
            precision * int(batch_size*seq_len*hidden_size / tensor_parallel))
        # (B x S x 4H) => (B x S x 4H)
        self.ffn_gelu = GeluLayer(OperatorCustomConfig(
            op_name="ffn_gelu"),
            compute_time_gelu,
            precision * int(batch_size*seq_len*4*hidden_size / tensor_parallel))
        self.ffn_gelu_backward = GeluLayerBackward(OperatorCustomConfig(
            op_name="ffn_gelu_backward"),
            compute_time_gelu_backward,
            precision * int(batch_size*seq_len*4*hidden_size / tensor_parallel))
        # (B x S x 4H) => (B x S x H)
        self.ffn_linear_2 = LinearLayerAdam(OperatorCustomConfig(
            op_name="ffn_linear_2"),
            compute_time_layer2,
            precision * int(batch_size*seq_len*hidden_size / tensor_parallel),
            precision * int(4 * (hidden_size**2)))
        self.ffn_linear_2_backward = LinearLayerBackwardAdam(OperatorCustomConfig(
            op_name="ffn_linear_2_backward"),
            compute_time_layer2_backward,
            precision * int(batch_size*seq_len*4*hidden_size / tensor_parallel))
        # (B x S x H) => (B x S x H)
        self.ffn_allreduce = AllReduceLayer(OperatorCustomConfig(
            op_name="ffn_allreduce"),
            compute_time_allreduce)
        self.ffn_allreduce_backward = AllReduceLayer(OperatorCustomConfig(
            op_name="ffn_allreduce_backward"),
            compute_time_allreduce_backward)
        # (B x S x H) => (B x S x H)
        self.ffn_dropout = DropoutLayer(OperatorCustomConfig(
            op_name="ffn_dropout"),
            compute_time_dropout,
            precision * int(batch_size*seq_len*hidden_size),
            inplace=True)
        self.ffn_dropout_backward = DropoutLayerBackward(OperatorCustomConfig(
            op_name="ffn_dropout_backward"),
            compute_time_dropout_backward,
            precision * int(batch_size*seq_len*hidden_size))
        
        # Linear1 -> Gelu -> Linear2 -> AllReduce -> Dropout
        self.ffn_linear_1.add_next(self.ffn_gelu)
        self.ffn_gelu.add_next(self.ffn_linear_2)
        self.ffn_linear_2.add_next(self.ffn_allreduce)
        self.ffn_allreduce.add_next(self.ffn_dropout)
        # MIDLE FORWARD AND BACKWARD
        self.ffn_dropout.add_next(self.ffn_dropout_backward)
        # Dropout-Backward -> AllReduce-Backward -> Linear2-Backward ->
        # Gelu-Backward -> Linear1-Backward
        self.ffn_dropout_backward.add_next(self.ffn_allreduce_backward)
        self.ffn_allreduce_backward.add_next(self.ffn_linear_2_backward)
        self.ffn_linear_2_backward.add_next(self.ffn_gelu_backward)
        self.ffn_gelu_backward.add_next(self.ffn_linear_1_backward)
        # linear1 output
        self.ffn_linear_1.add_next_bf_output_free(self.ffn_gelu_backward)
        # gelu output
        self.ffn_gelu.add_next_bf_output_free(self.ffn_linear_2_backward)
        # linear2 output
        self.ffn_linear_2.add_next_bf_output_free(self.ffn_dropout_backward)

        self.ffn_dropout_backward.add_next_bf_gradin_free(self.ffn_linear_2_backward)
        self.ffn_linear_2_backward.add_next_bf_gradin_free(self.ffn_gelu_backward)
        self.ffn_gelu_backward.add_next_bf_gradin_free(self.ffn_linear_1_backward)

        self._subop = self.ffn_linear_1
        self.init_weight = [self.ffn_linear_1.init_weight,
                            self.ffn_linear_2.init_weight] 

    def add_next(self, next_op):
        self.ffn_dropout.add_next(next_op)

    def add_next_backward(self, next_backward_op):
        self.ffn_linear_1.add_next(next_backward_op)
    
    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        self.ffn_linear_1.add_next_bf_gradin_free(next_op)

class EmbeddingLayer(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig, 
                 compute_time: int, 
                 output_size: int, 
                 weight_size: int):
        super().__init__(config)
        # Embedding
        self.init_weight = CudaMalloc(OperatorNonComputationalConfig(
            op_name="init_weight"), weight_size)
        # output tensor
        self.output_malloc = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_malloc"), output_size)
        # computation
        self.embedding = CommonComputational(OperatorComputationalConfig(
            op_name="embedding"), compute_time)
        # free output tensor
        self.output_free = CudaMalloc(OperatorNonComputationalConfig(
            op_name="output_free"), -output_size)
        # Init-Weight -> Output-Malloc -> Embedding -> Output-Free
        self.init_weight.add_next(self.output_malloc)
        self.output_malloc.add_next(self.embedding)
        self.embedding.add_next(self.output_free)

        self._subop = self.output_malloc

    def add_next(self, next_op):
        self.embedding.add_next(next_op)
        
    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        next_op.add_next(self.output_free)

class EmbeddingLayerBackwardAdam(AbstractOperator):
    def __init__(self, config: OperatorCustomConfig, 
                 compute_time: int, 
                 grad_in_size: int):
        super().__init__(config)
        # grad in tensor
        # self.grad_in_malloc = CudaMalloc(OperatorNonComputationalConfig(
        #     op_name="grad_in_malloc"), grad_in_size) # 16 bit grad_in
        # backward computation
        self.embedding_backward = \
            CommonComputational(OperatorComputationalConfig(
                op_name="embedding_backward"), compute_time)
        # free grad in tensor
        # self.grad_in_free = CudaMalloc(OperatorNonComputationalConfig(
        #     op_name="grad_in_free"), -grad_in_size)
        # Grad-In-Malloc -> Embedding-Backward -> Grad-In-Free
        # self.grad_in_malloc.add_next(self.embedding_backward)
        self.embedding_backward.add_next(self.grad_in_free)

        self._subop = self.embedding_backward

    def add_next(self, next_op):
        self.embedding_backward.add_next(next_op)

    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        # next_op.add_next(self.grad_in_free)
        pass

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
                 compute_time_att_allreduce_backward: int,
                 compute_time_att_dropout_backward: int,
                 
                 compute_time_ffn_layer1: int,
                 compute_time_ffn_gelu : int,
                 compute_time_ffn_layer2: int,
                 compute_time_ffn_allreduce: int,
                 compute_time_ffn_dropout: int,
                 
                 compute_time_ffn_layer1_backward: int,
                 compute_time_ffn_gelu_backward : int,
                 compute_time_ffn_layer2_backward: int,
                 compute_time_ffn_allreduce_backward: int,
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

        self.layernorm_1 = LayerNormLayer(OperatorCustomConfig(
            op_name="layernorm_1"),
            compute_time_layernorm_1,
            precision * int(batch_size*seq_len*hidden_size))
        self.layernorm_1_backward = LayerNormLayerBackward(OperatorCustomConfig(
            op_name="layernorm_1_backward"),
            compute_time_layernorm_1_backward,
            precision * int(batch_size*seq_len*hidden_size))

        self.core_attention = CoreAttentionTPRowParallel(OperatorCustomConfig(
            op_name="core_attention"),
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
            compute_time_att_allreduce_backward,
            compute_time_att_dropout_backward,

            batch_size,
            seq_len,
            head_num,
            head_hidden_size,
            tensor_parallel,
            precision)

        self.residual_add_1 = AddLayer(OperatorCustomConfig(
            op_name="residual_add_1"),
            compute_time_residual_add_1,
            precision * int(batch_size*seq_len*hidden_size))
        self.residual_add_1_backward = AddLayerBackward(OperatorCustomConfig(
            op_name="residual_add_1_backward"),
            compute_time_residual_add_1_backward,
            precision * int(batch_size*seq_len*hidden_size))
        
        self.layernorm_2 = LayerNormLayer(OperatorCustomConfig(
            op_name="layernorm_2"),
            compute_time_layernorm_2,
            precision * int(batch_size*seq_len*hidden_size))
        self.layernorm_2_backward = LayerNormLayerBackward(OperatorCustomConfig(
            op_name="layernorm_2_backward"),
            compute_time_layernorm_2_backward,
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
            compute_time_ffn_allreduce_backward,
            compute_time_ffn_dropout_backward,
            
            batch_size,
            seq_len,
            hidden_size,
            tensor_parallel,
            precision)

        self.residual_add_2 = AddLayer(OperatorCustomConfig(
            op_name="residual_add_2"),
            compute_time_residual_add_2,
            precision * int(batch_size*seq_len*hidden_size))
        self.residual_add_2_backward = AddLayerBackward(OperatorCustomConfig(
            op_name="residual_add_2_backward"),
            compute_time_residual_add_2_backward,
            precision * int(batch_size*seq_len*hidden_size))
        
        # layernorm_1 -> core_attention -> residual_add_1 -> 
        # layernorm_2 -> ffn_layer -> residual_add_2
        self.layernorm_1.add_next(self.core_attention)
        self.core_attention.add_next(self.residual_add_1)
        self.residual_add_1.add_next(self.layernorm_2)
        self.layernorm_2.add_next(self.ffn_layer)
        self.ffn_layer.add_next(self.residual_add_2)
        # MIDLE FORWARD AND BACKWARD
        self.residual_add_2.add_next(self.residual_add_2_backward)
        # last layer of ffn
        self.residual_add_2_backward.add_next_a(self.ffn_layer.ffn_dropout_backward)
        self.residual_add_2_backward.add_next_b(self.residual_add_1_backward)
        self.ffn_layer.add_next_backward(self.layernorm_2_backward)
        self.layernorm_2_backward.add_next(self.residual_add_1_backward)
        # last layer of core attention
        self.residual_add_1_backward.add_next_a(self.core_attention.output_dropout_backward)
        self.core_attention.add_next_backward(self.layernorm_1_backward)

        # layernorm_1 output should be free after core attention linear qkv backward
        self.layernorm_1.add_next_bf_output_free(self.core_attention.linear_qkv_backward)
        # residual_add_1 output
        self.residual_add_1.add_next_bf_output_free(self.layernorm_2_backward)
        # layernorm_2 output should be free after ffn linear 1 backward
        self.layernorm_2.add_next_bf_output_free(self.ffn_layer.ffn_linear_1_backward)
        
        # residual_add_2 backward
        self.residual_add_2_backward.add_next_bf_gradin_free(self.ffn_layer.ffn_dropout_backward)
        self.residual_add_2_backward.add_next_bf_gradin_free(self.residual_add_1_backward)
        # ffn_layer backward
        self.ffn_layer.add_next_bf_gradin_free(self.layernorm_2_backward)
        # layernorm_2 backward
        self.layernorm_2_backward.add_next_bf_gradin_free(self.residual_add_1_backward)
        # residual_add_1 backward
        self.residual_add_1_backward.add_next_bf_gradin_free(self.core_attention.output_dropout_backward)
        # core_attention backward
        self.core_attention.add_next_bf_gradin_free(self.layernorm_1_backward)
        
        self._subop = self.layernorm_1
        self.init_weight = [*self.core_attention.init_weight,
                            *self.ffn_layer.init_weight]
        
    def add_next(self, next_op):
        self.residual_add_2.add_next(next_op)
    
    def add_next_backward(self, next_backward_op):
        self.layernorm_1_backward.add_next(next_backward_op)
        self.residual_add_2_backward.add_next_b(next_backward_op)
    
    def add_next_bf_output_free(self, next_op):
        """
        append output tensor free operator to next_op
        """
        self.residual_add_2.add_next_bf_output_free(next_op)

    def add_next_bf_gradin_free(self, next_op):
        """
        append grad in tensor free operator to next_op
        """
        self.layernorm_1_backward.add_next_bf_gradin_free(next_op)
        self.residual_add_1_backward.add_next_bf_gradin_free(next_op)

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
                 compute_time_att_allreduce_backward: int,
                 compute_time_att_dropout_backward: int,
                 
                 compute_time_ffn_layer1: int,
                 compute_time_ffn_gelu : int,
                 compute_time_ffn_layer2: int,
                 compute_time_ffn_allreduce: int,
                 compute_time_ffn_dropout: int,
                 
                 compute_time_ffn_layer1_backward: int,
                 compute_time_ffn_gelu_backward : int,
                 compute_time_ffn_layer2_backward: int,
                 compute_time_ffn_allreduce_backward: int,
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
        self.embedding = EmbeddingLayer(OperatorCustomConfig(
            op_name="embedding"),
            compute_time_embedding,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(vocab_size*hidden_size))
        self.embedding_backward = EmbeddingLayerBackwardAdam(OperatorCustomConfig(
            op_name="embedding_backward"),
            compute_time_embedding_backward,
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
            compute_time_att_allreduce_backward,
            compute_time_att_dropout_backward,

            compute_time_ffn_layer1,
            compute_time_ffn_gelu,
            compute_time_ffn_layer2,
            compute_time_ffn_allreduce,
            compute_time_ffn_dropout,
            
            compute_time_ffn_layer1_backward,
            compute_time_ffn_gelu_backward,
            compute_time_ffn_layer2_backward,
            compute_time_ffn_allreduce_backward,
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
        
        self.layernorm_final = LayerNormLayer(OperatorCustomConfig(
            op_name="layernorm_final"),
            compute_time_layernorm_final,
            precision * int(batch_size*seq_len*hidden_size))
        
        self.layernorm_final_backward = LayerNormLayerBackward(OperatorCustomConfig(
            op_name="layernorm_final_backward"),
            compute_time_layernorm_final_backward,
            precision * int(batch_size*seq_len*hidden_size))
        
        self.embedding.add_next(self.transformer_blocks[0])
        self.transformer_blocks[0].add_next_backward(self.embedding_backward)
        self.transformer_blocks[0].add_next_bf_gradin_free(self.embedding_backward)
        for i in range(transformer_layers - 1):
            self.transformer_blocks[i].add_next(self.transformer_blocks[i + 1])
            self.transformer_blocks[i + 1].add_next_backward(self.transformer_blocks[i])
            self.transformer_blocks[i].add_next_bf_output_free(self.transformer_blocks[i + 1].layernorm_1_backward)
            self.transformer_blocks[i + 1].add_next_bf_gradin_free(self.transformer_blocks[i].residual_add_2_backward)
        self.transformer_blocks[-1].add_next(self.layernorm_final)
        self.transformer_blocks[-1].add_next_bf_output_free(self.layernorm_final_backward)
        self.layernorm_final_backward.add_next(self.transformer_blocks[-1].residual_add_2_backward)
        self.layernorm_final_backward.add_next_bf_gradin_free(self.transformer_blocks[-1].residual_add_2_backward)
        
        self.layernorm_final.add_next(self.layernorm_final_backward)
        self.layernorm_final.add_next_bf_output_free(self.layernorm_final_backward)